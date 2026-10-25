"""Extraction agent: read a news item or outage notice, return a validated structured signal.

Workflow (more than a single prompt):
  1. The model proposes a JSON record.
  2. Deterministic checks (schema.validate) test enums, units, dates, plausibility and that the quoted
     evidence really appears in the source text.
  3. If any check fails, the exact problems are sent back and the model must correct its answer
     (up to MAX_RETRIES times). Records that never pass are marked as failed, never silently kept.
"""
import json
import os
import re

from schema import validate

MAX_RETRIES = 2

SYSTEM = """You extract structured signals for a European power-market trading desk from short energy news items and outage notices.
Return ONLY one JSON object, no other text, with exactly these keys:
- event_type: one of plant_outage, grid_outage, gas_supply, weather, demand, policy, other
- fuel: one of gas, coal, lignite, nuclear, wind, solar, hydro, grid, none, other (grid = transmission lines and interconnectors)
- zone: two-letter country code (DE, FR, NL, GB, ...) if a country is stated or clearly implied, otherwise null
- capacity_mw: capacity unavailable or affected, in MW (convert GW to MW), only if the text gives or directly implies a figure in MW or GW; otherwise null. Never use percentages or gas volumes.
- status: planned, unplanned, resolved (the problem has ended) or unknown
- direction: expected effect on wholesale power prices according to the text: up, down or neutral
- start_time, end_time: 'YYYY-MM-DD' if the text states them, otherwise null. Never guess dates.
- confidence: your confidence from 0 to 1
- evidence: one short phrase copied EXACTLY from the text that supports your answer
If the item is not about market fundamentals, use event_type other, fuel none and direction neutral."""


def user_prompt(text: str) -> str:
    return f"Text:\n\"\"\"\n{text}\n\"\"\"\nReturn the JSON object."


def parse_json(raw: str):
    """Accept plain JSON or JSON wrapped in a code fence or surrounded by text."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


class AnthropicClient:
    """Thin wrapper around the Anthropic Python SDK (pip install anthropic).
    Reads ANTHROPIC_API_KEY from the environment. Model name: set LLM_MODEL (see docs.claude.com for current names)."""

    def __init__(self, model=None, max_tokens=600):
        import anthropic  # imported here so the tests run without the package

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("LLM_MODEL", "claude-haiku-5-5")
        self.max_tokens = max_tokens
        self.input_tokens = self.output_tokens = 0

    def complete(self, system, messages):
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages)
        self.input_tokens += r.usage.input_tokens
        self.output_tokens += r.usage.output_tokens
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class FakeClient:
    """Returns scripted answers in order. Used by the tests, no network needed."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), 0

    def complete(self, system, messages):
        self.calls += 1
        return self.answers[min(self.calls - 1, len(self.answers) - 1)]


def extract(text: str, client, max_retries: int = MAX_RETRIES) -> dict:
    """Returns {'ok', 'record', 'errors', 'attempts'}."""
    messages = [{"role": "user", "content": user_prompt(text)}]
    errors = []
    for attempt in range(1, max_retries + 2):
        raw = client.complete(SYSTEM, messages)
        try:
            rec = parse_json(raw)
            errors = validate(rec, text)
        except (json.JSONDecodeError, ValueError) as e:
            rec, errors = None, [f"Could not parse JSON: {e}"]
        if not errors:
            return {"ok": True, "record": rec, "errors": [], "attempts": attempt}
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": "Your answer had these problems:\n- " + "\n- ".join(errors)
                      + "\nReturn the corrected JSON object only."}]
    return {"ok": False, "record": None, "errors": errors, "attempts": max_retries + 1}
