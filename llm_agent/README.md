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

## Test set: `data/real_events_notices.jsonl` (38 notices)
- Each notice describes a **real event** (EDF and RWE outages, Norwegian gas outages, interconnector faults, wind-driven price
  moves, demand reports, EU carbon policy). The source link is in the `source` field.
- **Provenance, stated plainly:** the notices are written in the author's own words (an AI assistant, Claude), not copied
  verbatim from the sources, and the labels were also made by that assistant. This is **not an independent benchmark**: a
  language model may find text written in a similar style easier than real notices. Treat results as indicative.
- Mix: 13 plant outages, 4 grid outages, 5 gas supply, 6 weather, 4 demand, 3 policy, 3 other.
- Label rules: capacity_mw only if the text gives or directly implies MW or GW of capacity affected (not generation output,
  gas volumes or percentages); dates only if a full date with year is stated; zone = country of the asset (the EU is not a
  country; for interconnectors the first country named; null if none); status planned only if the text says planned,
  scheduled, maintenance or refuelling; direction = effect on wholesale power prices as stated or clearly implied.
- `data/labeled_notices.jsonl` holds 30 older **fictional** notices used only to test the code. Do not quote scores from it.

## Results so far
Keyword baseline on the 38 notices (written before this test set existed and not tuned on it), `results/agent_eval/agent_baseline_metrics.json`:
event type 68%, fuel 90%, zone 74%, status 84%, direction 66%, capacity 82%, dates 95%;
**all key fields (type, fuel, direction, capacity) correct: 39%**.
The LLM agent has **not been run yet** (it needs an API key and internet access, which the author's sandbox lacks).

## Run
- Tests (no key needed): `python tests/test_agent.py`
- Baseline: `python llm_agent/run_eval.py --method baseline --data llm_agent/data/real_events_notices.jsonl --out results/agent_eval`
- LLM agent: `pip install -r llm_agent/requirements.txt`, set `ANTHROPIC_API_KEY` in your terminal (never in a file you upload),
  optionally `LLM_MODEL` (check docs.claude.com for current model names), then
  `python llm_agent/run_eval.py --method llm --data llm_agent/data/real_events_notices.jsonl --out results/agent_eval`.
  The run prints token counts; cost depends on the model.

## Optional: add independent labels
For a stronger result, collect real notices yourself, label them by hand before running any model
(`templates/my_notices_template.csv`, then `python llm_agent/csv_to_jsonl.py my_notices.csv`), and evaluate on that file.
Keep copied notice texts on your own computer (they are in `.gitignore`).
