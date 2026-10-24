# LLM agent: structured signals from energy news and outage notices

Reads a short news item or outage notice and returns a validated record: event type, fuel, country, capacity (MW),
planned/unplanned status, start and end dates, expected price direction, and a quote from the text as evidence.

## How it works (more than one prompt)
1. The model proposes a JSON record (`agent.py`, called through the Anthropic API, not a chat window).
2. Deterministic checks (`schema.py`) test allowed values, units (GW converted to MW), plausible capacity, date format,
   end after start, and that the quoted evidence **really appears in the source text**.
3. If a check fails, the exact problems are sent back and the model must correct its answer (up to 2 retries).
   Records that never pass are counted as failures, never silently kept.
4. `aggregate.py` turns records into a daily table per country (unplanned thermal MW, planned MW, grid MW, net direction score).
5. `baseline.py` is a keyword and regex extractor that the agent has to beat; `evaluate.py` scores both field by field.

## Run
- Tests (no key needed): `python tests/test_agent.py`
- Baseline: `python llm_agent/run_eval.py --method baseline`
- LLM agent: `pip install -r llm_agent/requirements.txt`, set `ANTHROPIC_API_KEY` in your terminal (never in a file you upload),
  optionally `LLM_MODEL` (check docs.claude.com for current model names), then `python llm_agent/run_eval.py --method llm`.
  Cost depends on the model and the number of notices; the run prints token counts.

## Important: what the numbers mean
`data/labeled_notices.jsonl` holds 30 **synthetic** notices with invented plant names, written by the developer together
with the baseline rules. They only prove that the code works. The baseline scores 100% on fuel, zone and capacity there,
which is far too optimistic because the rules were written with these sentences in view. **Do not quote any accuracy figure
from this set.** For a real result: collect at least 30 real notices (for example public unavailability messages or news
snippets; check each source's terms of use), label them by hand **before** running any model, then report the baseline and the
LLM agent on that set. Label rules: direction = effect on wholesale power prices stated or clearly implied by the text;
capacity only if the text gives MW or GW; dates only if stated.

## Real evaluation, step by step
1. Open `templates/my_notices_template.csv` in a spreadsheet, format the `start_time` and `end_time` columns as Text, and save a copy
   as `my_notices.csv`. Collect at least 30 real notices, one per row (paste the notice text into `text`, the page address into `source`).
2. Fill in the answer columns **by hand, before running any model**. Allowed values: event_type = plant_outage, grid_outage,
   gas_supply, weather, demand, policy, other; fuel = gas, coal, lignite, nuclear, wind, solar, hydro, grid, none, other;
   status = planned, unplanned, resolved, unknown; direction = up, down, neutral (effect on wholesale power prices, as the text
   states or clearly implies). Leave zone, capacity_mw and the dates empty when the text does not give them.
3. `python llm_agent/csv_to_jsonl.py my_notices.csv` (checks every label and tells you what to fix).
4. `python llm_agent/run_eval.py --method baseline --data llm_agent/data/real_notices.jsonl`
5. `python llm_agent/run_eval.py --method llm --data llm_agent/data/real_notices.jsonl`
6. Compare `results/agent_baseline_metrics.json` and `results/agent_llm_metrics.json`. Keep the notice texts on your own computer
   (they are in `.gitignore`); publish only the results.
