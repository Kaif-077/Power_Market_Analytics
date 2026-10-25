"""Development experiments that led to the frozen settings in 02_final_evaluation.py.

Only the DEVELOPMENT period is used (walk-forward, 2025-04 to 2026-03). The final test period
(2026-04 to 2026-09) is not touched. Output: results/dev_experiments.csv

Note: HistGradientBoosting with loss='absolute_error' was tried first but was far too slow
(several minutes per configuration), so it was dropped and is not included here.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from pmp import config as C, data, evaluate as E, features as F

C.RESULTS_DIR.mkdir(exist_ok=True)
df = F.build_dataset(data.load_weather(), data.load_actuals(), data.load_entsoe_forecast())
fs = F.feature_sets(df)
slim = [c for c in fs["weather_plus_obs"]
        if c.startswith(("ws_", "wdsin_", "wdcos_", "gust_", "ws3_", "ramp", "doy_", "hour", "cap_proxy", "gen_"))]
folds = E.month_folds(C.DEV_START, C.TEST_START)
idx = df.index[(df.index >= pd.Timestamp(C.DEV_START, tz="UTC")) & (df.index < pd.Timestamp(C.TEST_START, tz="UTC"))]
y = df.loc[idx, "y"]
rows = []


def record(name, pred, secs=0.0):
    m = E.metrics(y, pred)
    rows.append({"experiment": name, **{k: round(v, 1) for k, v in m.items()}})
    print(f"{name:58s} MAE {m['MAE']:7.0f}  RMSE {m['RMSE']:7.0f}  bias {m['bias']:7.0f}  ({secs:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "dev_experiments.csv", index=False)


def run(name, cols, **kw):
    t = time.time()
    record(name, E.walk_forward_gbm(df, cols, folds, **kw), time.time() - t)


record("ENTSO-E benchmark", df.loc[idx, "entsoe_fc"])
record("persistence 48h", E.persistence_48h(df, idx))
record("power curve (isotonic)", E.walk_forward_power_curve(df, folds))

print("\n-- weather models: target scaling, features, recency", flush=True)
run("gbm weather+obs, raw target", fs["weather_plus_obs"], normalize=False, max_leaf_nodes=15)
run("gbm weather+obs, capacity-factor target", fs["weather_plus_obs"], normalize=True, max_leaf_nodes=15)
run("gbm weather+obs, cf target, recency half-life 180d", fs["weather_plus_obs"], normalize=True, max_leaf_nodes=15, half_life_days=180)
run("gbm slim features, cf target", slim, normalize=True, max_leaf_nodes=15)
run("gbm slim features, cf target, recency 180d", slim, normalize=True, max_leaf_nodes=15, half_life_days=180)
run("gbm weather-only, cf target", fs["weather_only"], normalize=True, max_leaf_nodes=15)

print("\n-- hybrid models that use the ENTSO-E forecast", flush=True)
run("hybrid direct (cf target)", fs["hybrid_with_entsoe"], normalize=True, max_leaf_nodes=15)
run("hybrid residual, leaves15 iter400 lr.05", fs["hybrid_with_entsoe"], base_col="entsoe_fc", max_leaf_nodes=15)
run("hybrid residual, leaves7 iter200 lr.03 msl100  [FROZEN]", fs["hybrid_with_entsoe"], base_col="entsoe_fc",
    max_leaf_nodes=7, max_iter=200, learning_rate=0.03, min_samples_leaf=100)
run("hybrid residual, leaves15 iter150 lr.03 msl100", fs["hybrid_with_entsoe"], base_col="entsoe_fc",
    max_leaf_nodes=15, max_iter=150, learning_rate=0.03, min_samples_leaf=100)
run("hybrid residual, leaves7 iter400 lr.02 msl200 l2=5", fs["hybrid_with_entsoe"], base_col="entsoe_fc",
    max_leaf_nodes=7, max_iter=400, learning_rate=0.02, min_samples_leaf=200, l2=5.0)
print("done", flush=True)
