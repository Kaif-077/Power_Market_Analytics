# European Power Market Analytics Platform (work in progress)

Day-ahead forecasting of German wind generation, built to mirror work done on an energy trading desk.
Status: data checks, the wind forecasting model and the battery back-test are done. Dashboard, pipeline automation and LLM agent are still to come.

## Run it
1. `pip install -r requirements.txt`
2. Put your raw files in `data/raw/` (not included): `entsoe/*.csv` (from `download_entsoe.py`),
   `weather/weather_previous_runs_day1.csv` (from `download_weather.py`), `smard/*.csv` (SMARD downloads),
   `fuels/` (TTF gas and EUA carbon daily CSV exports from Investing.com).
3. `python tests/test_no_leakage.py`        # checks no future data enters the features
4. `python scripts/01_dev_experiments.py`    # development experiments (about 7 minutes)
5. `python scripts/02_final_evaluation.py`   # final test, tables and figures (about 3 minutes)

6. `python tests/test_battery.py` and `python tests/test_price_no_leakage.py`   # checks for the back-test
7. `python scripts/03_price_model.py`        # walk-forward day-ahead price model (about 1.5 minutes)
8. `python scripts/04_backtest.py`           # battery back-test with the final test (about 1 minute);
   add `--dev-only` to see only the development period

Results are written to `results/`. Set `PMP_RAW_DIR` to read data from another folder.

## Method in one paragraph
Walk-forward validation: for each month the model is retrained on all earlier data (2-day gap) and
predicts that month. Settings were chosen on April 2025 to March 2026 only; April to September 2026
is a final test used once. Weather inputs are day-ahead forecasts (Open-Meteo Previous Runs API) at 8
German locations. Features from actual generation only use data older than the forecast cut-off.
The benchmark is ENTSO-E's own published day-ahead wind forecast.

## Back-test in one paragraph
A virtual 1 MW / 2 MWh battery trades the German day-ahead market: each day it charges in the hours predicted to be
cheapest and discharges in the hours predicted to be dearest (one full cycle per day, 88% round-trip efficiency,
2 EUR/MWh on every MWh traded), and is paid the actual hourly price. Prices are predicted from forecast residual load
(load minus wind minus solar), gas and carbon closes (only those dated two days before delivery or earlier) and
earlier prices. The variant was picked on April 2025 to March 2026; April to September 2026 is the final test.

## Known limitations
- The weather-only model is clearly worse than ENTSO-E; only the hybrid (ENTSO-E forecast plus a learned
  correction) beats it, by a modest margin.
- ENTSO-E's forecast may be published after the day-ahead auction closes; check before claiming this
  model could have been used live.
- Weather forecasts are one model's "day before" run, with the model chosen automatically by the API.
- Back-test: most of the profit comes from the daily price shape (solar dip, evening peak), which even naive
  forecasts capture (94% of the perfect-foresight profit vs 98% for the model). The Sharpe ratio is not a meaningful risk
  measure here: every day is profitable. Real risks are shrinking spreads as storage grows, 15-minute products
  (SMARD gives hourly averages), outages, and costs not modelled (investment, grid fees, market impact).
- 2026 spreads were unusually wide after the March 2026 gas price spike; do not extrapolate the profit per day.

## Dashboard
`streamlit run app.py` (needs only the files in `results/`, not the raw data).
Deploy free: push the repo to GitHub, then on share.streamlit.io choose the repo, branch `main` and file `app.py`.
