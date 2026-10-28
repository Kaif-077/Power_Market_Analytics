"""Convert your hand-labelled spreadsheet (CSV) into the file the evaluation reads, checking every label.

  python llm_agent/csv_to_jsonl.py my_notices.csv
Writes llm_agent/data/real_notices.jsonl (kept out of GitHub by .gitignore). Fix any reported problem and run again.
"""
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from schema import DIRECTIONS, EVENT_TYPES, FUELS, STATUSES  # noqa: E402

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
src = Path(sys.argv[1])
rows = list(csv.DictReader(open(src, encoding="utf-8-sig", newline="")))
out, problems = [], []
for n, r in enumerate(rows, start=2):  # spreadsheet row number
    r = {k.strip(): (v or "").strip() for k, v in r.items()}
    if not r.get("text"):
        continue
    if r.get("id") == "0":  # the example row in the template
        continue
    g = {"event_type": r["event_type"], "fuel": r["fuel"], "zone": r["zone"] or None, "status": r["status"],
         "direction": r["direction"], "start_time": r["start_time"] or None, "end_time": r["end_time"] or None}
    for key, allowed in (("event_type", EVENT_TYPES), ("fuel", FUELS), ("status", STATUSES), ("direction", DIRECTIONS)):
        if g[key] not in allowed:
            problems.append(f"row {n}: {key} '{g[key]}' must be one of {allowed}")
    if g["zone"] and not re.fullmatch(r"[A-Z]{2}", g["zone"]):
        problems.append(f"row {n}: zone '{g['zone']}' must be two capital letters like DE, or empty")
    for k in ("start_time", "end_time"):
        if g[k] and not DATE.match(g[k]):
            problems.append(f"row {n}: {k} '{g[k]}' must look like 2026-10-05 (format the column as Text before typing)")
    try:
        g["capacity_mw"] = float(r["capacity_mw"].replace(",", "")) if r["capacity_mw"] else None
    except ValueError:
        problems.append(f"row {n}: capacity_mw '{r['capacity_mw']}' must be a plain number in MW, or empty")
        g["capacity_mw"] = None
    out.append({"id": len(out) + 1, "synthetic": False, "source": r.get("source", ""), "text": r["text"], "gold": g})
if problems:
    print("Please fix these and run again:\n" + "\n".join(problems))
    sys.exit(1)
dest = HERE / "data" / "real_notices.jsonl"
with open(dest, "w", encoding="utf-8") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(f"OK: {len(out)} labelled notices written to {dest}")
if len(out) < 30:
    print("Note: fewer than 30 notices; results will be too uncertain to quote.")
