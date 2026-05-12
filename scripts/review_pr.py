import os
from google import genai
from google.genai import types
from github import Github
import github


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


def main():
    gh = Github(auth=github.Auth.Token(os.environ["GITHUB_TOKEN"]))
    repo = gh.get_repo(os.environ["REPO_NAME"])
    pr = repo.get_pull(int(os.environ["PR_NUMBER"]))

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(api_version="v1beta"),
    )

    files = list(pr.get_files())
    base_ref = pr.base.ref

    print(f"Reviewing PR #{pr.number}: '{pr.title}'")
    print(f"  Base branch:   {base_ref}")
    print(f"  Files changed: {len(files)}")

    system_prompt = load_system_prompt()
    prompt = build_prompt(pr, files, repo, base_ref)
    full_prompt = f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n---\n\n{prompt}"

    print(f"  Prompt size:   ~{len(full_prompt) // 4:,} tokens")
    print("Sending to Gemini...")

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=8200,
        ),
    )

    review_text = response.text
    pr.create_review(body=review_text, event="COMMENT")
    print("Review posted successfully.")


if __name__ == "__main__":
    main()
