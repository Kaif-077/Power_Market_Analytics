"""Walk-forward day-ahead price model. Writes results/price_predictions.csv.

For every month from 2024-04 to 2026-09 the model is retrained on all earlier data (2-day gap) and
predicts that month. Two variants are produced:
  level : predicts the hourly price directly
  shape : predicts price minus yesterday's daily mean (then adds it back)
Naive baselines (same hour yesterday / last week) are included. Nothing is tuned here; the choice of
variant is made on the development period only, in 04_backtest.py.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pmp import config as C, data, evaluate as E, price_features as PF
from pmp.models import make_gbm

C.RESULTS_DIR.mkdir(exist_ok=True)
df = PF.build_price_dataset(data.load_da_prices(), data.load_fundamental_forecasts(), data.load_fuel_prices())
feats = PF.FEATURES
FIRST_TEST_MONTH, END = "2024-04-01", C.TEST_END
folds = E.month_folds(FIRST_TEST_MONTH, END)
print(f"{len(df)} hours, {len(folds)} monthly folds", flush=True)

PARAMS = dict(max_leaf_nodes=15, max_iter=300, learning_rate=0.05, min_samples_leaf=40)
out = pd.DataFrame({"price": df["price"], "local_date": df["local_date"], "n_hours": df["n_hours_in_day"]})
out["naive_yesterday"], out["naive_lastweek"] = df["p_lag24"], df["p_lag168"]

for name in ("level", "shape"):
    t = time.time()
    target = df["price"] - df["p_yday_mean"] if name == "shape" else df["price"]
    preds = []
    for ms, me in folds:
        tr = (df.index < ms - pd.Timedelta(days=C.EMBARGO_DAYS)) & target.notna() & df[feats].notna().any(axis=1)
        te = (df.index >= ms) & (df.index < me)
        m = make_gbm(**PARAMS).fit(df.loc[tr, feats], target[tr])
        p = pd.Series(m.predict(df.loc[te, feats]), index=df.index[te])
        preds.append(p + df.loc[te, "p_yday_mean"] if name == "shape" else p)
    out[f"model_{name}"] = pd.concat(preds)
    print(f"variant {name} done in {time.time() - t:.0f}s", flush=True)

out = out.loc[out["model_level"].notna()]
out.to_csv(C.RESULTS_DIR / "price_predictions.csv")
dev = (out.index >= pd.Timestamp(C.DEV_START, tz="UTC")) & (out.index < pd.Timestamp(C.TEST_START, tz="UTC"))
print("\nDEVELOPMENT-period price accuracy only (EUR/MWh):")
for c in ("naive_yesterday", "naive_lastweek", "model_level", "model_shape"):
    print(f"  {c:16s}", {k: round(v, 1) for k, v in E.metrics(out.loc[dev, 'price'], out.loc[dev, c]).items() if k in ("MAE", "RMSE", "bias")})
