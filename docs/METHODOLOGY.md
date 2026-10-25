# Methodology

## 1. Question
Can a model forecast German wind generation and day-ahead power prices well enough to beat simple benchmarks, and is a
price-based trading rule (a virtual battery) worth more with a model than with naive guesses? Everything is evaluated the way
it would have worked in practice: only information available before the day-ahead auction (12:00 CET on D-1) may be used.

## 2. Data (all hourly, converted to UTC)
| Data | Source | Notes |
|---|---|---|
| Wind, solar, load, prices | SMARD (Bundesnetzagentur) | German number format; local time with 23- and 25-hour days, converted to UTC |
| Day-ahead forecasts and actual generation | ENTSO-E Transparency Platform | benchmark forecast; actual wind and solar were split across columns after July 2025, merged |
| Weather forecasts | Open-Meteo Previous Runs API | the forecast issued one day earlier, at 8 German locations (onshore, offshore, south) |
| Gas (TTF) and carbon (EUA) | Investing.com exports | only the close dated two days before delivery or earlier is used |
Checks: no duplicates or gaps, realistic ranges, SMARD and ENTSO-E actuals agree within 0.5%, partial first and last days dropped.

## 3. Validation design
- **Walk-forward:** each month is predicted by a model trained only on earlier data, with a 2-day gap.
- **Development period** (April 2025 to March 2026) was used to choose settings; **final test** (April to September 2026) was used once.
- **Leakage tests** scramble future data and check that no feature changes. They also cover the clock-change days.

## 4. Wind forecast
- **Baselines:** persistence (same hour 2 days earlier), a power curve (isotonic regression of capacity factor on mean wind speed).
- **Weather model:** gradient boosting on forecast wind speed, direction, gusts, pressure and temperature at 8 locations, regional means,
  wind speed cubed, recent changes, time features, a yearly capacity proxy and recent actual generation (stopping at the cut-off).
- **Hybrid:** predicts ENTSO-E's error (actual minus its forecast) and adds it back. Strong regularisation: it is a small correction.
- **Result (test, 4,392 hours):** ENTSO-E MAE 1,387 MW; hybrid 1,323 MW (-4.6%, bootstrap interval for the difference -117 to -19 MW);
  weather-only model 2,182 MW (+57%). The 80% interval covered 75% of outcomes.

## 5. Price model and battery back-test
- **Price model:** gradient boosting on forecast residual load (load minus wind minus solar), gas and carbon costs, calendar and earlier prices.
  Test error 21 EUR/MWh against 33 (copy yesterday) and 42 (copy last week).
- **Battery:** 1 MW / 2 MWh, one cycle a day, 88% round-trip efficiency, 2 EUR/MWh on each MWh traded. A dynamic program picks charge and
  discharge hours from predicted prices; profit is booked at actual prices.
- **Result (test, 181 days):** model EUR 53,934 (98% of perfect foresight); copy-yesterday EUR 51,809 (94%). Sensitivity table in `results/`.

## 6. Business impact (illustrative)
No imbalance prices were available, so each MWh of forecast error is assumed to cost 5 to 30 EUR (`results/business_impact.csv`).
At 10 EUR/MWh the hybrid's smaller errors are worth about EUR 2.8 million over the six test months across the whole German wind fleet,
and the weather-only model would cost about EUR 35 million more than ENTSO-E. These are scenarios, not measured savings.

## 7. LLM agent
Described in `llm_agent/README.md`: extraction with validation and retries, a keyword baseline (39% of notices with all four key fields right),
and a test set written and labelled by an AI assistant. The agent itself has not been run yet.

## 8. Limitations
Weather-only forecasts are weaker than ENTSO-E's. ENTSO-E's forecast may be published after the auction closes. The weather input is one forecast
product at 8 points. Most battery profit comes from the daily price shape that naive forecasts also capture. Hourly averages hide 15-minute
prices. 2026 spreads were unusually wide. Costs, grid fees, wear and market impact are simplified or ignored.

## 9. Questions to be ready for
1. **Why not a random train/test split?** It would train on the future and overstate accuracy.
2. **How do you know there is no leakage?** A fixed information cut-off plus tests that scramble future data.
3. **Did you beat the benchmark?** Only with the hybrid, by 4.6%; the weather-only model did not.
4. **Why does the hybrid work?** Correcting a good forecast is easier than rebuilding it.
5. **Is the battery profit real?** It is a back-test with stated assumptions; most of it comes from the predictable daily price shape.
6. **What would you do next?** Live daily forecasts with a public scoreboard, imbalance price data, and a real-notice evaluation of the agent.
