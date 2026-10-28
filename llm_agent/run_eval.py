"""Evaluate the baseline or the LLM agent on the hand-labelled notices.

  python llm_agent/run_eval.py --method baseline
  python llm_agent/run_eval.py --method llm --provider gemini      (pip install google-genai; GEMINI_API_KEY set)
  python llm_agent/run_eval.py --method llm --provider anthropic   (pip install anthropic; ANTHROPIC_API_KEY set)
Writes agent_<baseline | llm_anthropic | llm_gemini>_metrics.json, _predictions.jsonl and _signals.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agent import AnthropicClient, GeminiClient, extract  # noqa: E402
from aggregate import build_signals  # noqa: E402
from baseline import extract_baseline  # noqa: E402
from evaluate import grounded_rate, score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["baseline", "llm"], required=True)
ap.add_argument("--provider", choices=["anthropic", "gemini"], default="anthropic", help="which AI service to call for --method llm")
ap.add_argument("--data", default=str(HERE / "data" / "labeled_notices.jsonl"))
ap.add_argument("--out", default=str(HERE.parent / "results"))
a = ap.parse_args()

items = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
texts, gold = [i["text"] for i in items], [i["gold"] for i in items]
pred, attempts, failures = [], [], []
extra = {}
if a.method == "baseline":
    pred = [extract_baseline(t) for t in texts]
else:
    client = GeminiClient() if a.provider == "gemini" else AnthropicClient()
    for i, t in enumerate(texts):
        try:
            r = extract(t, client)
        except Exception as e:  # network or API error: record it and carry on
            r = {"ok": False, "record": None, "errors": [f"API error: {e}"], "attempts": 0}
            time.sleep(2)
        pred.append(r["record"]); attempts.append(r["attempts"])
        if not r["ok"]:
            failures.append({"id": items[i]["id"], "errors": r["errors"]})
        print(f"{i + 1}/{len(texts)} ok={r['ok']} attempts={r['attempts']}", flush=True)
    extra = {"provider": a.provider, "model": client.model, "input_tokens": client.input_tokens, "output_tokens": client.output_tokens,
             "share_needing_retry": round(sum(x > 1 for x in attempts) / len(attempts), 3),
             "failed_validation": len(failures), "failures": failures}

m = score(gold, pred)
m["grounded_evidence_rate"] = grounded_rate(texts, pred)
m.update(extra)
tag = "baseline" if a.method == "baseline" else f"llm_{a.provider}"
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
json.dump(m, open(out / f"agent_{tag}_metrics.json", "w"), indent=1)
with open(out / f"agent_{tag}_predictions.jsonl", "w", encoding="utf-8") as f:
    for i, p in zip(items, pred):
        f.write(json.dumps({"id": i["id"], "pred": p}) + "\n")
sig = build_signals(pred)
sig.round(2).to_csv(out / f"agent_{tag}_signals.csv", index=False)
print(json.dumps({k: v for k, v in m.items() if k != "failures"}, indent=1))
