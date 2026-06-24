import os
import re
import requests
from google import genai
from google.genai import types
from github import Github


def get_file_context(repo, filename: str, ref: str) -> str:
    try:
        content = repo.get_contents(filename, ref=ref)
        return content.decoded_content.decode("utf-8", errors="replace")
    except Exception as e:
        return f"(could not fetch: {e})"


def build_prompt(pr, files, repo, base_ref: str) -> str:
    sections = [f"## PR: {pr.title}", ""]

    if pr.body:
        sections += [f"**Description:** {pr.body.strip()}", ""]

    sections += ["---", "## Changed files (diff)", ""]

    for f in files:
        sections.append(f"### `{f.filename}` — {f.status}")

        if f.patch:
            sections.append(f"```diff\n{f.patch}\n```")
        else:
            sections.append("_(binary or empty patch)_")

        if f.status in ("modified", "renamed"):
            ctx = get_file_context(repo, f.filename, base_ref)
            sections += [
                "",
                f"**Full file before changes (`{base_ref}`):**",
                f"```\n{ctx}\n```",
            ]

        sections.append("")

    return "\n".join(sections)


def load_system_prompt(path: str = "prompts/system_prompt.txt") -> str:
    try:
        with open(path, "r") as fh:
            return fh.read()
    except FileNotFoundError:
        raise SystemExit(f"Error: system prompt not found at '{path}'.")


# Gemini backend
def review_with_gemini(system_prompt: str, prompt: str) -> str:
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(api_version="v1"),
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(
            role="user",
            parts=[types.Part(text=f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n---\n\n{prompt}")],
        )],
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=8200),
    )
    return response.text


# Local fine-tuned backend
LOCAL_SYSTEM = (
    "You are a senior code reviewer. Given a code diff, write ONE short review "
    "comment (1-2 sentences) about the single most important issue. Be specific."
)


def _clean(text: str) -> str:
    """Strip code fences / suggestion blocks so the comment renders as plain text."""
    text = text.replace("```suggestion", "").replace("```", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def review_with_local(files) -> str:
    endpoint = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
    sections = []
    for f in files:
        if not f.patch:
            continue
        # split the file patch into individual @@ hunks (training distribution)
        hunks = re.split(r'(?=^@@ )', f.patch, flags=re.M)
        hunks = [h for h in hunks if h.strip().startswith("@@")]
        if not hunks:
            hunks = [f.patch]

        comments = []
        for hunk in hunks[:6]:                    # cap to avoid spamming huge PRs
            r = requests.post(endpoint, json={
                "model": "pr-reviewer",
                "system": LOCAL_SYSTEM,
                "prompt": f"Review this change:\n```diff\n{hunk.strip()}\n```",
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 80, "repeat_penalty": 1.2},
            })
            r.raise_for_status()
            c = _clean(r.json().get("response", ""))
            if len(c) >= 5:
                comments.append(f"- {c}")

        body = "\n".join(comments) if comments else "_(model flagged no specific issue)_"
        sections.append(f"### `{f.filename}`\n{body}")
    return "\n\n".join(sections) if sections else "_No reviewable diffs found._"


def main():
    gh = Github(os.environ["GITHUB_TOKEN"])
    repo = gh.get_repo(os.environ["REPO_NAME"])
    pr = repo.get_pull(int(os.environ["PR_NUMBER"]))

    files = list(pr.get_files())
    base_ref = pr.base.ref
    backend = os.environ.get("BACKEND", "gemini")

    print(f"Reviewing PR #{pr.number}: '{pr.title}'")
    print(f"  Base branch:   {base_ref}")
    print(f"  Files changed: {len(files)}")
    print(f"  Backend:       {backend}")

    if backend == "local":
        print("Reviewing with local fine-tuned model (Ollama)...")
        review_text = review_with_local(files)
    else:
        system_prompt = load_system_prompt()
        prompt = build_prompt(pr, files, repo, base_ref)
        print(f"  Prompt size:   ~{len(prompt) // 4:,} tokens")
        print("Sending to Gemini...")
        review_text = review_with_gemini(system_prompt, prompt)

    pr.create_review(body=review_text, event="COMMENT")
    print("Review posted successfully.")


if __name__ == "__main__":
    main()