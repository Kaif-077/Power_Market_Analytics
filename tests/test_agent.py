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
assert validate({**GOOD, "end_time": "2027-03-15"}, TEXT)  # year 2027 is not in the text: a guess
assert validate({**GOOD, "end_time": "2026-03-15"}, TEXT) == []  # year 2026 is in the text
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

# Gemini wrapper: tested against a stand-in for the google-genai package (no network, no key)
import types as _t

_google, _genai, _gtypes = _t.ModuleType("google"), _t.ModuleType("google.genai"), _t.ModuleType("google.genai.types")
class _Obj:
    def __init__(self, **kw): self.__dict__.update(kw)
_gtypes.Content = lambda role, parts: _Obj(role=role, parts=parts)
_gtypes.Part = lambda text: _Obj(text=text)
_gtypes.GenerateContentConfig = lambda **kw: _Obj(**kw)
_seen = {}
class _Models:
    def generate_content(self, model, contents, config):
        _seen.update(model=model, roles=[c.role for c in contents], system=config.system_instruction)
        return _Obj(text="{}", usage_metadata=_Obj(prompt_token_count=7, candidates_token_count=3))
_genai.Client = lambda: _Obj(models=_Models())
_genai.types, _google.genai = _gtypes, _genai
sys.modules.update({"google": _google, "google.genai": _genai, "google.genai.types": _gtypes})
from agent import GeminiClient
g = GeminiClient(model="test-model", min_seconds_between_calls=0)
out = g.complete("SYS", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}])
assert out == "{}" and _seen["roles"] == ["user", "model", "user"] and _seen["system"] == "SYS" and _seen["model"] == "test-model"
assert (g.input_tokens, g.output_tokens) == (7, 3)
print("OK: agent validation, retry loop, scoring, baseline and aggregation behave as expected")
