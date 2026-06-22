"""Chat-templating. Training, eval, and inference all build prompts from here so
they can never drift. If they drift, eval numbers are lying."""

import json
from config import SYSTEM_PROMPT


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def _user(diff_hunk: str) -> str:
    """The single source of truth for the user-turn wording."""
    return f"Review this change:\n```diff\n{diff_hunk}\n```"


def build_messages(diff_hunk: str, body: str | None = None):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user(diff_hunk)},
    ]
    if body is not None:
        msgs.append({"role": "assistant", "content": body})
    return msgs


def to_text(example, tokenizer):
    """Dataset row -> {'text': full chat string} for SFTTrainer (Phase 3)."""
    msgs = build_messages(example["diff_hunk"], example["body"])
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=False)}


def build_inference_prompt(diff_hunk: str, tokenizer):
    """Prompt-only string ending with the assistant cue — for base/tuned models."""
    msgs = build_messages(diff_hunk, body=None)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# --- few-shot "strong prompt" baseline -------------------------------------
# Loaded from TRAIN (never test -> no leakage). train.jsonl is already shuffled
# with a fixed seed, so the first N short examples are deterministic.
def load_fewshot(n=3, path="data/train.jsonl"):
    rows = load_jsonl(path)
    picks = [r for r in rows if len(r["body"]) < 200 and len(r["diff_hunk"]) < 600]
    return picks[:n]


def build_fewshot_prompt(diff_hunk: str, tokenizer, shots):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in shots:
        msgs.append({"role": "user", "content": _user(ex["diff_hunk"])})
        msgs.append({"role": "assistant", "content": ex["body"]})
    msgs.append({"role": "user", "content": _user(diff_hunk)})
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)