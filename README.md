# PR Review Bot 

Automatically reviews your pull requests using Gemini AI and posts structured feedback as a comment — triggered on every push.

---

## How it works

1. You open or push to a PR
2. GitHub Actions kicks off
3. Gemini reads the diff + the full original file for context
4. A structured review gets posted as a comment on your PR

---

## Setup

**1. Add your Gemini API key to GitHub Secrets**
```
Settings → Secrets and variables → Actions → New repository secret
Name: GEMINI_API_KEY
```

**2. Enable GitHub Actions**

The workflow lives at `.github/workflows/pr-review.yml` — it runs automatically on every PR.

**3. Open a PR and watch it work**

---

## What the review looks like

```
## Summary
What the PR does in 2-3 sentences.

## Issues
[CRITICAL] sql_handler.py:42 — SQL injection via unsanitized input. Fix: use parameterized queries.
[HIGH]     auth.py:17 — MD5 used for token generation. Fix: use secrets.token_hex().

## Suggestions
- auth.py:30 — DB connection never closed, use try/finally or a context manager.

## Verdict
REQUEST CHANGES — two critical security issues must be resolved before merging.
```

---

## Stack

- **Gemini 2.5 Flash** — does the actual review
- **PyGithub** — fetches the diff and posts the comment
- **GitHub Actions** — orchestrates everything

---

## Project structure

```
.
├── .github/
│   └── workflows/
│       └── pr-review.yml
├── scripts/
│   └── review_pr.py
├── prompts/
│   └── system_prompt.txt
└── requirements.txt
```
---

## Limitations

- Free tier Gemini has a daily quota check that before using
- Binary files are skipped
- Very large PRs with more files may hit token limits