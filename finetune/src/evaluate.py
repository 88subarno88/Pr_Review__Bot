import os, re, json, time, requests
from rouge_score import rouge_scorer
import sacrebleu

OLLAMA = "http://localhost:11434/api/generate"
TUNED_MODEL = "pr-reviewer"
BASE_MODEL  = "qwen2.5-coder:3b"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
JUDGE_MODEL = "qwen2.5:7b"           # local judge 
TEST_PATH = "data/test.jsonl"
JUDGE_N = 60
CACHE = "results/preds_cache.json"

SYSTEM = ("You are a senior code reviewer. Given a code diff, write ONE short review "
          "comment (1-2 sentences) about the single most important issue. Be specific.")

def load(p): return [json.loads(l) for l in open(p)]
def user_turn(diff): return f"Review this change:\n```diff\n{diff}\n```"
def clean(t): return re.sub(r"\s+", " ", t.replace("```suggestion","").replace("```","")).strip()

# persistent cache
def load_cache():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    return {"preds": {}, "judge": {}, "latency": {}}

def save_cache(c):
    os.makedirs("results", exist_ok=True)
    tmp = CACHE + ".tmp"
    json.dump(c, open(tmp, "w"))
    os.replace(tmp, CACHE)

# generators
def ollama_gen(model, diff):
    r = requests.post(OLLAMA, json={
        "model": model, "system": SYSTEM, "prompt": user_turn(diff),
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 100, "repeat_penalty": 1.2},
    })
    r.raise_for_status()
    return clean(r.json().get("response", ""))

def gemini_call(system, user, max_tokens=120, retries=4):
    url = (f"https://generativelanguage.googleapis.com/v1/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}")
    payload = {"contents":[{"parts":[{"text":f"{system}\n\n{user}"}]}],
               "generationConfig":{"temperature":0.2,"maxOutputTokens":max_tokens,
                                   "thinkingConfig":{"thinkingBudget":0}}}
    for a in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=60)
            if r.status_code in (429,503,500):
                w = 8*(2**a); print(f"  gemini {r.status_code}, wait {w}s"); time.sleep(w); continue
            r.raise_for_status()
            data = r.json()
            cand = data.get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", [])
            text = parts[0].get("text", "") if parts else ""
            time.sleep(5)
            return text.strip()
        except Exception as e:
            print("  gemini err:", str(e)[:80]); time.sleep(5)
    return ""
def gemini_gen(diff):
    return clean(gemini_call(SYSTEM, user_turn(diff)))

# metrics 
SC = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
def task_metrics(preds, refs):
    rl = sum(SC.score(r,p)["rougeL"].fmeasure for p,r in zip(preds,refs))/len(preds)
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    return round(rl,4), round(bleu,2)

# LOCAL judge (qwen2.5:7b via Ollama) 
JUDGE_SYS = ("You evaluate code-review comments. Given a DIFF and a CANDIDATE review, "
             "rate how useful and correct the candidate is from 1 (useless/wrong) to "
             "5 (sharp, correct, actionable). A human REFERENCE is one valid example; "
             "the candidate may be worded differently and still be excellent. "
             "Reply with ONLY a single digit 1-5 and nothing else.")
def judge_one(diff, cand, ref):
    if len(cand) < 3:
        return 1
    prompt = (f"DIFF:\n{diff}\n\nHUMAN REFERENCE:\n{ref}\n\n"
              f"CANDIDATE REVIEW:\n{cand}\n\nScore (1-5):")
    r = requests.post(OLLAMA, json={
        "model": JUDGE_MODEL, "system": JUDGE_SYS, "prompt": prompt,
        "stream": False, "options": {"temperature": 0.0, "num_predict": 5},
    })
    r.raise_for_status()
    out = r.json().get("response", "")
    m = re.findall(r"[1-5]", out)
    return int(m[-1]) if m else None

# generation with per-item caching
def gen_local_cached(name, model, diffs, cache):
    done = cache["preds"].get(name, [])
    if len(done) >= len(diffs):
        print(f"{name}: already cached ({len(done)})"); return done
    print(f"\n=== generating {name} (resuming at {len(done)}/{len(diffs)}) ===")
    for i in range(len(done), len(diffs)):
        done.append(ollama_gen(model, diffs[i]))
        if (i+1) % 20 == 0:
            cache["preds"][name] = done; save_cache(cache); print(f"  {i+1}/{len(diffs)}")
    cache["preds"][name] = done; save_cache(cache)
    return done

def gen_gemini_cached(diffs, judge_idx, cache):
    done = cache["preds"].get("gemini", [])
    if len(done) >= len(judge_idx):
        print(f"gemini: already cached ({len(done)})"); return done
    print(f"\n=== generating gemini (resuming at {len(done)}/{len(judge_idx)}) ===")
    for k in range(len(done), len(judge_idx)):
        done.append(gemini_gen(diffs[judge_idx[k]]))
        cache["preds"]["gemini"] = done; save_cache(cache)
        print(f"  gemini {k+1}/{len(judge_idx)}")
    return done

# judging with per-item caching 
def judge_cached(name, diffs, preds, refs, judge_idx, cache):
    done = cache["judge"].get(name, [])
    targets = judge_idx if name != "gemini" else list(range(len(judge_idx)))
    if len(done) >= len(targets):
        print(f"judge[{name}]: already cached ({len(done)})"); return done
    print(f"=== judging {name} (resuming at {len(done)}/{len(targets)}) ===")
    for k in range(len(done), len(targets)):
        if name == "gemini":
            d, p, r = diffs[judge_idx[k]], preds[k], refs[judge_idx[k]]
        else:
            idx = judge_idx[k]; d, p, r = diffs[idx], preds[idx], refs[idx]
        done.append(judge_one(d, p, r))
        cache["judge"][name] = done; save_cache(cache)
        print(f"  judge[{name}] {k+1}/{len(targets)} -> {done[-1]}")
    return done

def main():
    test = load(TEST_PATH)
    diffs = [t["diff_hunk"] for t in test]
    refs  = [t["body"] for t in test]
    judge_idx = list(range(min(JUDGE_N, len(test))))
    cache = load_cache()
    print(f"test rows: {len(test)} | judged subset: {len(judge_idx)}")

    t0 = time.time(); gen_local_cached("base", BASE_MODEL, diffs, cache)
    cache["latency"]["base"] = cache["latency"].get("base") or round((time.time()-t0)/len(diffs),3)
    t0 = time.time(); gen_local_cached("tuned", TUNED_MODEL, diffs, cache)
    cache["latency"]["tuned"] = cache["latency"].get("tuned") or round((time.time()-t0)/len(diffs),3)
    gen_gemini_cached(diffs, judge_idx, cache)
    save_cache(cache)

    for name in ["base", "tuned", "gemini"]:
        judge_cached(name, diffs, cache["preds"][name], refs, judge_idx, cache)
    save_cache(cache)

    results = {}
    for name in ["base", "tuned", "gemini"]:
        p = cache["preds"][name]
        if name == "gemini":
            rl, bleu = task_metrics(p, [refs[i] for i in judge_idx])
        else:
            rl, bleu = task_metrics(p, refs)
        js = [s for s in cache["judge"].get(name, []) if s is not None]
        results[name] = {"rougeL":rl, "bleu":bleu,
                         "judge": round(sum(js)/len(js),2) if js else None,
                         "judge_n": len(js), "latency_s": cache["latency"].get(name)}
        print(f"{name}: rougeL={rl} bleu={bleu} judge={results[name]['judge']} (n={len(js)})")

    json.dump(results, open("results/eval_scores.json","w"), indent=2)
    write_md(results, len(test), len(judge_idx))
    print("\nwrote results/comparison.md + eval_scores.json")

def write_md(r, n_all, n_judge):
    cost = {"base":"$0 (local)","tuned":"$0 (local)","gemini":"~API cost"}
    L = ["# Code-Review Model Comparison (held-out test set)\n",
         f"ROUGE-L / BLEU over all {n_all} rows (Gemini over {n_judge}) · LLM-judge "
         f"(qwen2.5:7b, local) over {n_judge} rows.\n",
         "| Model | ROUGE-L | BLEU | Judge (1-5) | Latency/req | Cost |",
         "|---|---|---|---|---|---|"]
    for k in ["base","tuned","gemini"]:
        m=r[k]; L.append(f"| {k} | {m['rougeL']} | {m['bleu']} | {m['judge']} "
                         f"| {m['latency_s']}s | {cost[k]} |")
    L += ["","## Notes",
          "- ROUGE/BLEU are weak proxies for free-form review; the judge score is the metric that matters.",
          "- base and tuned are the same 3B model; the delta is the fine-tuning effect.",
          "- Judged locally with qwen2.5:7b for full reproducibility (no API dependency).",
          "- Gemini is the frontier baseline (whole-PR context vs per-hunk for the local models)."]
    open("results/comparison.md","w").write("\n".join(L))

if __name__ == "__main__":
    main()