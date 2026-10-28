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

## Results so far (38 notices, one run each; test set written and labelled by an AI assistant, so indicative only)
| Field correct | Keyword baseline | Gemini agent (gemini-3.1-flash-lite) |
|---|---|---|
| **All four key fields** (type, fuel, direction, capacity) | **39%** | **89%** (roughly 76% to 96% given only 38 notices) |
| Event type | 68% | 97% |
| Fuel | 90% | 97% |
| Direction | 66% | 95% |
| Capacity (MW) | 82% | 95% |
| Country | 74% | 87% |
| Status (planned/unplanned) | 84% | 66% |
| Start / end date | 95% / 95% | 76% / 71% |

Other measures for the agent: every evidence quote was found in the source text (100%), no answer needed a retry, none failed validation,
13,327 input and 4,283 output tokens in total. Files: `results/agent_eval/agent_llm_gemini_*` and `agent_baseline_*`.

**Where it is weaker, from the predictions file:** 18 of the 20 date misses are the model inventing a year (for example 2024) when the notice
gave none, although the instructions say never to guess. 13 status misses are mostly the model choosing planned or unplanned when the notice
did not say, and some of those labels are open to argument. It returned no country for 5 notices where the asset's country is implied, and it
treated 2 policy items as price-lowering. It also took 700 MW of oil-fired output and a 12 GW tender as capacity. ### Second run, after the date fix (not a clean test)
After the first run, a date check was added to `schema.py`: a date is rejected, and the model asked to correct it, when its year does not appear in the text.
It was added after seeing the errors on this same test set, so these numbers are not a clean test; the first-run table above stays the clean result.

| Field correct | First run (clean) | After date fix |
|---|---|---|
| Start date | 76% | **92%** |
| End date | 71% | **100%** |
| All four key fields | 89% (34 of 38) | 87% (33 of 38) |
| Fuel | 97% | 95% |
| Notices needing a retry | 0% | 34% |
| Tokens (input / output) | 13,327 / 4,283 | 20,438 / 5,664 |

The date fix did what it was meant to: the invented years disappeared. The one-notice drop in key fields is not caused by dates; it is the model
labelling a Shell wind-farm sale as wind instead of none in the second run only, which is ordinary run-to-run variation, so differences of one notice
should not be read as real. Status (66%) is unchanged. Files: `results/agent_eval_v2/`.

## Run
- Tests (no key needed): `python tests/test_agent.py`
- Baseline: `python llm_agent/run_eval.py --method baseline --data llm_agent/data/real_events_notices.jsonl --out results/agent_eval`
- LLM agent with Google Gemini (free tier available): `pip install google-genai`, then set the key **in your terminal only**
  (Windows: `set GEMINI_API_KEY=your_key`), optionally `set GEMINI_MODEL=model_name` (check the current name in Google AI Studio), then
  `python llm_agent/run_eval.py --method llm --provider gemini --data llm_agent/data/real_events_notices.jsonl --out results/agent_eval`.
- LLM agent with Anthropic: `pip install anthropic`, set `ANTHROPIC_API_KEY`, optionally `LLM_MODEL`, and use `--provider anthropic`.
  Calls are slowed and retried to respect rate limits. Never put a key in a file you upload or paste it into a chat.

## Optional: add independent labels
For a stronger result, collect real notices yourself, label them by hand before running any model
(`templates/my_notices_template.csv`, then `python llm_agent/csv_to_jsonl.py my_notices.csv`), and evaluate on that file.
Keep copied notice texts on your own computer (they are in `.gitignore`).
