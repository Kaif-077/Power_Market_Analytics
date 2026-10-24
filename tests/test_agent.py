"""Tests for the extraction agent (no network, no API key). Run with:  python tests/test_agent.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_agent"))

from agent import FakeClient, extract, parse_json
from aggregate import build_signals
from baseline import extract_baseline
from evaluate import score
from schema import validate

TEXT = "Unit 3 of the Westhafen lignite power plant in Germany (660 MW) tripped unexpectedly on 12 March 2026."
GOOD = {"event_type": "plant_outage", "fuel": "lignite", "zone": "DE", "capacity_mw": 660, "status": "unplanned",
        "direction": "up", "start_time": "2026-03-12", "end_time": None, "confidence": 0.9,
        "evidence": "tripped unexpectedly on 12 March 2026"}

# validation catches each kind of problem
assert validate(GOOD, TEXT) == []
assert validate({**GOOD, "fuel": "oil"}, TEXT)
assert validate({**GOOD, "capacity_mw": 660000}, TEXT)
assert validate({**GOOD, "capacity_mw": "660 MW"}, TEXT)
assert validate({**GOOD, "end_time": "2026-03-01"}, TEXT)
assert validate({**GOOD, "start_time": "12/03/2026"}, TEXT)
assert validate({**GOOD, "evidence": "reactor exploded"}, TEXT)  # quote not in the text
assert validate({k: v for k, v in GOOD.items() if k != "zone"}, TEXT)

# JSON in a code fence is accepted
assert parse_json("```json\n" + json.dumps(GOOD) + "\n```")["fuel"] == "lignite"

# agent: valid first time / fixed after feedback / never valid
r = extract(TEXT, FakeClient([json.dumps(GOOD)]))
assert r["ok"] and r["attempts"] == 1
bad = {**GOOD, "evidence": "reactor exploded", "capacity_mw": -5}
c = FakeClient([json.dumps(bad), json.dumps(GOOD)])
r = extract(TEXT, c)
assert r["ok"] and r["attempts"] == 2 and c.calls == 2
r = extract(TEXT, FakeClient(["not json at all"]), max_retries=2)
assert not r["ok"] and r["attempts"] == 3 and r["record"] is None

# scoring
gold = {k: GOOD[k] for k in ("event_type", "fuel", "zone", "capacity_mw", "status", "direction", "start_time", "end_time")}
s = score([gold, gold], [GOOD, None])
assert s["event_type"] == 0.5 and s["answered"] == 1 and s["key_fields_all_correct"] == 0.5

# baseline and aggregation
b = extract_baseline(TEXT)
assert b["fuel"] == "lignite" and b["capacity_mw"] == 660 and b["zone"] == "DE" and b["start_time"] == "2026-03-12"
sig = build_signals([GOOD, None, {**GOOD, "event_type": "other"}])
assert len(sig) == 1 and sig.loc[0, "unplanned_thermal_mw"] == 660
print("OK: agent validation, retry loop, scoring, baseline and aggregation behave as expected")
