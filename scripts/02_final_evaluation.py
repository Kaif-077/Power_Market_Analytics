"""Final evaluation. Settings were frozen from the development folds (see results/dev_grid.csv
and the notes in the README). The test period 2026-04 to 2026-09 is used once, here."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmp import config as C, data, evaluate as E, features as F

C.RESULTS_DIR.mkdir(exist_ok=True)
C.FIG_DIR.mkdir(exist_ok=True)

# ---- frozen settings -------------------------------------------------------------------------
WEATHER_PARAMS = dict(normalize=True, max_leaf_nodes=15)  # weather-only / weather+obs models
HYBRID_PARAMS = dict(base_col="entsoe_fc", max_leaf_nodes=7, max_iter=200,
                     learning_rate=0.03, min_samples_leaf=100)  # correction of the ENTSO-E forecast

df = F.build_dataset(data.load_weather(), data.load_actuals(), data.load_entsoe_forecast())
fs = F.feature_sets(df)


def run_period(start, end, label):
    folds = E.month_folds(start, end)
    idx = df.index[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index < pd.Timestamp(end, tz="UTC"))]
    y = df.loc[idx, "y"]
    preds = {
        "entsoe_benchmark": df.loc[idx, "entsoe_fc"],
        "persistence_48h": E.persistence_48h(df, idx),
        "power_curve": E.walk_forward_power_curve(df, folds),
        "gbm_weather_only": E.walk_forward_gbm(df, fs["weather_only"], folds, **WEATHER_PARAMS),
        "gbm_weather_plus_obs": E.walk_forward_gbm(df, fs["weather_plus_obs"], folds, **WEATHER_PARAMS),
        "hybrid_entsoe_correction": E.walk_forward_gbm(df, fs["hybrid_with_entsoe"], folds, **HYBRID_PARAMS),
    }
    table = pd.DataFrame({k: E.metrics(y, v) for k, v in preds.items()}).T
    table["MAE_vs_entsoe_pct"] = (table["MAE"] / table.loc["entsoe_benchmark", "MAE"] - 1) * 100
    table["RMSE_vs_entsoe_pct"] = (table["RMSE"] / table.loc["entsoe_benchmark", "RMSE"] - 1) * 100
    table.round(1).to_csv(C.RESULTS_DIR / f"metrics_{label}.csv")
    print(f"\n== {label} ({start} to {end}), {len(idx)} hours")
    print(table[["MAE", "RMSE", "bias", "MAE_pct_of_mean", "MAE_vs_entsoe_pct", "RMSE_vs_entsoe_pct"]].round(1).to_string())
    return folds, idx, y, preds


run_period(C.DEV_START, C.TEST_START, "dev")
folds, idx, y, preds = run_period(C.TEST_START, C.TEST_END, "test")

# ---- uncertainty: 10% and 90% quantiles of the hybrid correction ------------------------------
q = {}
for a in (0.1, 0.9):
    q[a] = E.walk_forward_gbm(df, fs["hybrid_with_entsoe"], folds, loss="quantile", quantile=a, **HYBRID_PARAMS)
lo, hi = np.minimum(q[0.1], q[0.9]), np.maximum(q[0.1], q[0.9])
coverage = float(((y >= lo) & (y <= hi)).mean())
width = float((hi - lo).mean())
print(f"\n80% interval (q10-q90): coverage {coverage * 100:.1f}% (target 80%), mean width {width:.0f} MW")

# ---- significance: day-block bootstrap vs ENTSO-E ---------------------------------------------
boot = {k: E.day_block_bootstrap(y, preds[k], preds["entsoe_benchmark"])
        for k in ("hybrid_entsoe_correction", "gbm_weather_only", "gbm_weather_plus_obs")}
print("\nDay-block bootstrap, MAE difference vs ENTSO-E (negative = better):")
print(json.dumps(boot, indent=1))
json.dump({"bootstrap_vs_entsoe": boot, "interval_80": {"coverage": coverage, "mean_width_MW": width}},
          open(C.RESULTS_DIR / "test_extras.json", "w"), indent=1)

# ---- breakdowns -------------------------------------------------------------------------------
out = pd.DataFrame({k: v for k, v in preds.items()})
out["actual"] = y
out["hybrid_q10"], out["hybrid_q90"] = lo, hi
out.round(1).to_csv(C.RESULTS_DIR / "predictions_test.csv")

ae = lambda k: (out[k] - out["actual"]).abs()
by_month = pd.DataFrame({k: ae(k).groupby(out.index.strftime("%Y-%m")).mean()
                         for k in ("entsoe_benchmark", "gbm_weather_plus_obs", "hybrid_entsoe_correction")})
by_hour = pd.DataFrame({k: ae(k).groupby(out.index.hour).mean()
                        for k in ("entsoe_benchmark", "gbm_weather_plus_obs", "hybrid_entsoe_correction")})
level = pd.qcut(out["actual"], 3, labels=["low wind", "medium wind", "high wind"])
by_level = pd.DataFrame({k: ae(k).groupby(level, observed=True).mean()
                         for k in ("entsoe_benchmark", "gbm_weather_plus_obs", "hybrid_entsoe_correction")})
by_month.round(0).to_csv(C.RESULTS_DIR / "test_mae_by_month.csv")
by_hour.round(0).to_csv(C.RESULTS_DIR / "test_mae_by_hour.csv")
by_level.round(0).to_csv(C.RESULTS_DIR / "test_mae_by_wind_level.csv")
print("\nTest MAE by month:\n", by_month.round(0).to_string())
print("\nTest MAE by wind level:\n", by_level.round(0).to_string())

# ---- figures ----------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
by_month.plot.bar(ax=ax, rot=0)
ax.set_ylabel("MAE (MW)"); ax.set_title("Test period: mean absolute error by month"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(C.FIG_DIR / "mae_by_month.png", dpi=130); plt.close(fig)

worst = ae("entsoe_benchmark").groupby(out.index.floor("D")).mean().rolling(7).mean().idxmax()
w0 = worst - pd.Timedelta(days=6); sl = out.loc[w0:worst + pd.Timedelta(days=1)]
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(sl.index, sl["actual"], "k", lw=1.6, label="actual")
ax.plot(sl.index, sl["entsoe_benchmark"], lw=1, label="ENTSO-E day-ahead")
ax.plot(sl.index, sl["hybrid_entsoe_correction"], lw=1, label="hybrid (ENTSO-E + correction)")
ax.plot(sl.index, sl["gbm_weather_plus_obs"], lw=1, ls="--", label="weather-only model")
ax.fill_between(sl.index, sl["hybrid_q10"], sl["hybrid_q90"], alpha=0.15, label="hybrid 80% interval")
ax.set_ylabel("MW"); ax.set_title("Hardest week in the test period for the ENTSO-E forecast")
ax.legend(fontsize=7, ncol=2); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(C.FIG_DIR / "hardest_week.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
by_level.plot.bar(ax=ax, rot=0)
ax.set_ylabel("MAE (MW)"); ax.set_title("Test period: error by wind generation level"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(C.FIG_DIR / "mae_by_wind_level.png", dpi=130); plt.close(fig)
print("\nsaved results and figures")
