# Code-Review Model Comparison (held-out test set)

ROUGE-L / BLEU over all 142 rows (Gemini over 20) · LLM-judge (qwen2.5:7b, local) over 20 rows.

| Model | ROUGE-L | BLEU | Judge (1-5) | Latency/req | Cost |
|---|---|---|---|---|---|
| base | 0.0991 | 1.41 | 3.2 | 0.0s | $0 (local) |
| tuned | 0.0901 | 2.38 | 2.9 | 0.0s | $0 (local) |
| gemini | 0.0613 | 0.0 | 2.9 | Nones | ~API cost |

## Notes
- ROUGE/BLEU are weak proxies for free-form review; the judge score is the metric that matters.
- base and tuned are the same 3B model; the delta is the fine-tuning effect.
- Judged locally with qwen2.5:7b for full reproducibility (no API dependency).
- Gemini is the frontier baseline (whole-PR context vs per-hunk for the local models).