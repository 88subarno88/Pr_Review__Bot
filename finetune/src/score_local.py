import json, re, os, requests
from rouge_score import rouge_scorer
import sacrebleu

OLLAMA="http://localhost:11434/api/generate"
SYSTEM=("You are a senior code reviewer. Given a code diff, write ONE short review "
        "comment (1-2 sentences) about the single most important issue. Be specific.")
def clean(t): return re.sub(r"\s+"," ",t.replace("```suggestion","").replace("```","")).strip()
def gen(model,d):
    r=requests.post(OLLAMA,json={"model":model,"system":SYSTEM,
        "prompt":f"Review this change:\n```diff\n{d}\n```","stream":False,
        "options":{"temperature":0.2,"num_predict":100,"repeat_penalty":1.2}})
    return clean(r.json().get("response",""))

test=[json.loads(l) for l in open("data/test.jsonl")]
diffs=[t["diff_hunk"] for t in test]; refs=[t["body"] for t in test]
SC=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True)

preds={}
for name,model in [("base","qwen2.5-coder:3b"),("tuned","pr-reviewer")]:
    print(f"generating {name}...")
    preds[name]=[gen(model,d) for d in diffs]
    p=preds[name]
    rl=sum(SC.score(r,x)["rougeL"].fmeasure for x,r in zip(p,refs))/len(p)
    bleu=sacrebleu.corpus_bleu(p,[refs]).score
    print(f"{name}: rougeL={round(rl,4)} bleu={round(bleu,2)}")

os.makedirs("results",exist_ok=True)
json.dump({"preds":preds,"judge":{},"latency":{"base":None,"tuned":None}},
          open("results/preds_cache.json","w"))
print("cached base+tuned -> results/preds_cache.json")