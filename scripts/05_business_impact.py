"""Illustrative euro value of wind forecast errors on the final test period.

We have no imbalance or intraday price data, so this is a scenario table, not a measured saving: every MWh of
forecast error is assumed to cost a fixed amount (5 to 30 EUR/MWh) to correct. Output: results/business_impact.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pmp import config as C

p = pd.read_csv(C.RESULTS_DIR / "predictions_test.csv", index_col=0, parse_dates=True)
COSTS = [5, 10, 20, 30]  # assumed EUR per MWh of forecast error
models = {"entsoe_benchmark": "ENTSO-E day-ahead", "hybrid_entsoe_correction": "Hybrid (ENTSO-E + correction)",
          "gbm_weather_plus_obs": "Weather-based model"}
rows = []
for k, name in models.items():
    m = p[[k, "actual"]].dropna()
    err_mwh = float((m[k] - m["actual"]).abs().sum())  # hourly MW = MWh per hour
    row = {"model": name, "hours": len(m), "abs_error_GWh": round(err_mwh / 1000, 1),
           "abs_error_MWh_per_day": round(err_mwh / (len(m) / 24), 0)}
    for c in COSTS:
        row[f"cost_at_{c}_EUR_per_MWh_MEUR"] = round(err_mwh * c / 1e6, 2)
    rows.append(row)
t = pd.DataFrame(rows)
base = t.iloc[0]
for i in range(1, len(t)):  # difference to the benchmark (negative = cheaper)
    for c in COSTS:
        t.loc[i, f"vs_entsoe_at_{c}_MEUR"] = round(t.loc[i, f"cost_at_{c}_EUR_per_MWh_MEUR"] - base[f"cost_at_{c}_EUR_per_MWh_MEUR"], 2)
t.to_csv(C.RESULTS_DIR / "business_impact.csv", index=False)
pd.set_option("display.width", 250)
print(t.T.to_string(header=False))
