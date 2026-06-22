import os, re, json, time, random, requests
from dotenv import load_dotenv

load_dotenv()                       # reads finetune/.env for github token

GH = "https://api.github.com"
TOKEN = os.environ["GITHUB_TOKEN"]
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

REPOS = [
    ("facebook", "react"),
    ("vercel", "next.js"),
    ("pytorch", "pytorch"),
    # if these stay noisy, try: ("rust-lang", "rust"), ("kubernetes", "kubernetes")
]

OUT_DIR = "data"
TARGET_PAIRS = 2000
PER_REPO_CAP = 600          # so one repo's review style doesn't dominate

KNOWN_BOTS = {"dependabot", "codecov", "sonarcloud", "github-actions",
              "renovate", "greenkeeper", "vercel", "netlify"}

# non-actionable acknowledgements (author/reviewer chatter, not a review)
ACK = re.compile(
    r"^\s*(lgtm|looks good|makes sense|good (idea|catch|point)|nice|thanks|thank you|"
    r"done|fixed|reverted|addressed|agreed|sounds good|will do|yep|yup|yeah|ok(ay)?|"
    r"\+1|ditto|same as|got it|sg|sgtm)\b", re.I)

# comments that depend on context NOT present in the hunk
OFF_HUNK = re.compile(
    r"\b(addressed in|fixed in|see (the )?(other|previous|follow.?up)|"
    r"as (mentioned|discussed) (above|below)|same as (above|below))\b", re.I)


# fetch
def fetch_review_comments(owner, repo, max_pages=15):
    out = []
    for page in range(1, max_pages + 1):
        r = requests.get(
            f"{GH}/repos/{owner}/{repo}/pulls/comments",
            headers=HEADERS,
            params={"per_page": 100, "page": page,
                    "sort": "created", "direction": "desc"},
        )
        # graceful rate-limit handling
        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - int(time.time()), 1)
            print(f"  rate-limited, sleeping {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        time.sleep(0.7)
    return out


#  filter
def is_bot(comment) -> bool:
    user = comment.get("user") or {}
    login = (user.get("login") or "").lower()
    if user.get("type") == "Bot":
        return True
    if login.endswith("[bot]"):
        return True
    return any(b in login for b in KNOWN_BOTS)


def clean_body(body: str) -> str:
    if not body:
        return ""
    body = re.sub(r"@\w[\w-]*", "", body)             # @mentions
    body = re.sub(r"http\S+", "", body)               # URLs
    body = re.sub(r"^\s*>.*$", "", body, flags=re.M)  # quoted reply lines
    body = re.sub(r"\s+", " ", body)                  # collapse whitespace
    return body.strip()


def good_pair(diff_hunk: str, body: str) -> bool:
    if not diff_hunk.strip() or len(diff_hunk) > 1500:
        return False
    if len(body) < 20 or len(body) > 400:             # raised floor 15 -> 20
        return False
    if not re.search(r"[a-zA-Z]", body):              # nothing left after cleaning
        return False
    if ":robot:" in body or body.startswith("🤖"):    # AI-bot replies
        return False
    if ACK.match(body):                               # non-actionable ack
        return False
    if OFF_HUNK.search(body):                          # needs missing context
        return False
    return True


#  split
def split_and_write(pairs, seed=42):
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_test, n_val = int(0.10 * n), int(0.10 * n)
    test, val, train = pairs[:n_test], pairs[n_test:n_test + n_val], pairs[n_test + n_val:]
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        with open(f"{OUT_DIR}/{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  (test is SACRED)")


#  main
def main():
    seen, pairs = set(), []
    for owner, repo in REPOS:
        print(f"fetching {owner}/{repo} ...")
        raw = fetch_review_comments(owner, repo)
        kept = 0
        for c in raw:
            if kept >= PER_REPO_CAP:
                break
            if is_bot(c):
                continue
            if c.get("in_reply_to_id"):       # thread reply, not an original review
                continue
            hunk = c.get("diff_hunk") or ""
            body = clean_body(c.get("body") or "")
            if hunk in seen or not good_pair(hunk, body):
                continue
            seen.add(hunk)
            pairs.append({"diff_hunk": hunk, "body": body})
            kept += 1
        print(f"  kept {kept} from {owner}/{repo}")

    pairs = pairs[:TARGET_PAIRS]
    print(f"\nTOTAL clean pairs: {len(pairs)}")

    # >>> MANUAL QC — read these before trusting the data <
    for ex in random.sample(pairs, min(30, len(pairs))):
        print("\n--- DIFF ---\n", ex["diff_hunk"][:300])
        print("--- COMMENT ---\n", ex["body"])

    split_and_write(pairs)


if __name__ == "__main__":
    main()