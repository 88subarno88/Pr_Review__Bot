import os, re, time, json, argparse
from dotenv import load_dotenv
load_dotenv()

import config as C
from format import (load_jsonl, build_inference_prompt,
                    build_fewshot_prompt, load_fewshot)


# HF generators
def make_hf_generator(adapter_dir=None):
    from unsloth import FastLanguageModel
    import torch
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=C.MODEL_ID, max_seq_length=C.MAX_SEQ_LEN,
        load_in_4bit=True, dtype=None,
    )
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    FastLanguageModel.for_inference(model)          # 2x faster inference
    shots = load_fewshot(n=3)

    def gen(diff, fewshot=False):
        prompt = (build_fewshot_prompt(diff, tokenizer, shots) if fewshot
                  else build_inference_prompt(diff, tokenizer))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:],
                                skip_special_tokens=True)
        return text.strip()
    return gen


#  Gemini (gen + judge)
def _gemini():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(api_version="v1"))
    return client, types


def _gemini_call(client, types, system, user, max_tokens=128, retries=6):
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=C.GEMINI_MODEL,
                contents=[types.Content(role="user", parts=[types.Part(
                    text=f"SYSTEM INSTRUCTIONS:\n{system}\n\n---\n\n{user}")])],
                config=types.GenerateContentConfig(temperature=0.2,
                                                   max_output_tokens=max_tokens),
            )
            time.sleep(7)                     # gentle throttle for free-tier RPM
            return (resp.text or "").strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper():
                wait = 5 * (2 ** attempt)
                print(f"  rate-limited, backing off {wait}s...")
                time.sleep(wait)
                continue
            raise
    return ""


def make_gemini_generator():
    client, types = _gemini()
    def gen(diff, fewshot=False):
        return _gemini_call(client, types, C.SYSTEM_PROMPT,
                            f"Review this change:\n```diff\n{diff}\n```")
    return gen


JUDGE_SYS = (
    "You evaluate code-review comments. Given a diff and a CANDIDATE review, score how "
    "useful and correct the candidate is as a review of that diff, 1 (useless/wrong) to "
    "5 (sharp, correct, actionable). A human reference is given as ONE example of a valid "
    "review; the candidate may be worded differently and still be excellent. Judge the "
    "candidate on its own merit. Reply with ONLY the integer 1-5."
)


def make_judge():
    client, types = _gemini()
    def judge(diff, candidate, reference):
        if not candidate.strip():
            return 1
        user = (f"DIFF:\n```diff\n{diff}\n```\n\nHUMAN REFERENCE:\n{reference}\n\n"
                f"CANDIDATE:\n{candidate}\n\nScore (1-5):")
        out = _gemini_call(client, types, JUDGE_SYS, user, max_tokens=4)
        m = re.search(r"[1-5]", out)
        return int(m.group()) if m else 1
    return judge


# metrics
def task_metrics(preds, refs):
    from rouge_score import rouge_scorer
    import sacrebleu
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl = sum(sc.score(r, p)["rougeL"].fmeasure
             for p, r in zip(preds, refs)) / max(len(preds), 1)
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    return {"rougeL": round(rl, 4), "bleu": round(bleu, 2)}


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["base", "prompt", "gemini"])
    ap.add_argument("--adapter", default=C.ADAPTER_DIR)
    ap.add_argument("--limit", type=int, default=0, help="first N test rows (0=all)")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-judge (saves quota)")
    args = ap.parse_args()

    test = load_jsonl("data/test.jsonl")
    if args.limit:
        test = test[:args.limit]
    diffs = [t["diff_hunk"] for t in test]
    refs = [t["body"] for t in test]
    print(f"evaluating on {len(test)} test rows")

    # build only the generators we need (lazy -> gemini-only needs no GPU stack)
    hf = make_hf_generator(None) if ({"base", "prompt"} & set(args.models)) else None
    tuned = make_hf_generator(args.adapter) if "tuned" in args.models else None
    gem = make_gemini_generator() if "gemini" in args.models else None
    judge = None if args.no_judge else make_judge()

    plan = {"base": (hf, False), "prompt": (hf, True),
            "tuned": (tuned, False), "gemini": (gem, False)}

    results, samples = {}, []
    for name in args.models:
        gen, fewshot = plan[name]
        print(f"\n=== {name} ===")
        t0, preds = time.time(), []
        for i, d in enumerate(diffs):
            preds.append(gen(d, fewshot=fewshot))
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(diffs)}")
        latency = (time.time() - t0) / max(len(diffs), 1)

        m = task_metrics(preds, refs)
        if judge:
            scores = [judge(d, p, r) for d, p, r in zip(diffs, preds, refs)]
            m["judge"] = round(sum(scores) / len(scores), 2)
        m["latency_s_per_req"] = round(latency, 3)
        m["cost_per_1k_usd"] = (0.0 if name != "gemini" else C.GEMINI_COST_PER_1K)
        results[name] = m
        print(name, "->", m)

        # stash first 5 predictions for the README side-by-side
        for d, p in list(zip(diffs, preds))[:5]:
            samples.append({"model": name, "diff": d[:300], "pred": p})

    is_baseline = "tuned" not in args.models
    out = f"{C.RESULTS_DIR}/{'baseline' if is_baseline else 'tuned'}_scores.json"
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    json.dump(samples, open(f"{C.RESULTS_DIR}/sample_predictions.json", "w"), indent=2)
    write_comparison_md(results)
    print(f"\nwrote {out} + comparison.md")


def write_comparison_md(results):
    lines = ["# Base vs Prompt vs Tuned vs Gemini (held-out test set)\n",
             "| Model | ROUGE-L | BLEU | Judge (1-5) | Latency/req | Cost/1k |",
             "|---|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['rougeL']} | {m['bleu']} | {m.get('judge','-')} "
                     f"| {m['latency_s_per_req']}s | {m['cost_per_1k_usd']} |")
    lines.append("\n> ROUGE/BLEU are weak proxies for review quality; the judge score "
                 "is the metric that matters. Headline goes here after Phase 4.")
    open(f"{C.RESULTS_DIR}/comparison.md", "w").write("\n".join(lines))


if __name__ == "__main__":
    main()