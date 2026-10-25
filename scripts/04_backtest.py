"""Battery back-test on the walk-forward price predictions.

  python scripts/04_backtest.py --dev-only   # development period only (used to pick the price-model variant)
  python scripts/04_backtest.py              # adds the final out-of-sample test (2026-04 to 2026-09)

Assumptions (stated up front, tested in the sensitivity table): 1 MW / 2 MWh battery, one full cycle per
day, round-trip efficiency 88%, 2 EUR/MWh cost on every MWh bought and sold, price taker (no market impact),
profit booked at the hourly day-ahead price.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmp import battery as B, config as C, evaluate as E

ETA, COST = 0.88, 2.0
FROZEN_VARIANT = "level"  # chosen on the development table: higher profit, simpler (the two variants are statistically tied)
DEV_ONLY = "--dev-only" in sys.argv

P = pd.read_csv(C.RESULTS_DIR / "price_predictions.csv", index_col=0, parse_dates=True)
P.index = pd.to_datetime(P.index, utc=True)
P["local_date"] = pd.to_datetime(P["local_date"])
P["perfect"] = P["price"]


def window(df, start, end):
    return df[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index < pd.Timestamp(end, tz="UTC"))]


def table(df, cols, eta=ETA, cost=COST):
    pnl = {c: B.daily_pnl(df, c, eta, cost) for c in cols}
    rows = {c: B.perf(pnl[c], pnl["perfect"]) for c in cols}
    return pd.DataFrame(rows).T, pnl


dev = window(P, C.DEV_START, C.TEST_START)
cols = ["perfect", "model_level", "model_shape", "naive_yesterday", "naive_lastweek"]
dev_tab, _ = table(dev, cols)
dev_tab.round(2).to_csv(C.RESULTS_DIR / "backtest_dev.csv")
print(f"DEVELOPMENT period {C.DEV_START} to {C.TEST_START}  (eta={ETA}, cost={COST} EUR/MWh)")
print(dev_tab[["days", "total_EUR", "mean_EUR_per_day", "sharpe_annualised", "max_drawdown_EUR", "capture_of_perfect_foresight"]].round(2).to_string())
if DEV_ONLY or FROZEN_VARIANT is None:
    sys.exit(0)

# ------------------------------------------------------------------------------------------ final test
M = f"model_{FROZEN_VARIANT}"
test = window(P, C.TEST_START, C.TEST_END)
cols = ["perfect", M, "naive_yesterday", "naive_lastweek"]
test_tab, pnl = table(test, cols)
test_tab.round(2).to_csv(C.RESULTS_DIR / "backtest_test.csv")
print(f"\nFINAL TEST {C.TEST_START} to {C.TEST_END}  (model variant: {FROZEN_VARIANT})")
print(test_tab[["days", "total_EUR", "mean_EUR_per_day", "std_EUR_per_day", "sharpe_annualised", "max_drawdown_EUR",
                "worst_day_EUR", "share_days_profitable", "capture_of_perfect_foresight"]].round(2).to_string())

acc = pd.DataFrame({c: E.metrics(test["price"], test[c]) for c in (M, "naive_yesterday", "naive_lastweek")}).T
acc.round(1).to_csv(C.RESULTS_DIR / "price_accuracy_test.csv")
print("\nHourly price accuracy on test (EUR/MWh):\n", acc[["MAE", "RMSE", "bias"]].round(1).to_string())

extras = {
    "model_daily_profit_ci": B.bootstrap(pnl[M]),
    "model_minus_naive_yesterday_ci": B.bootstrap(pnl[M], pnl["naive_yesterday"]),
    "model_minus_naive_lastweek_ci": B.bootstrap(pnl[M], pnl["naive_lastweek"]),
}
print("\nBootstrap (days):", json.dumps(extras, indent=1))

# sensitivity to the assumptions
rows = []
for eta in (0.80, 0.85, 0.88, 0.92):
    for cost in (0.0, 1.0, 2.0, 4.0):
        t, _ = table(test, ["perfect", M], eta, cost)
        rows.append({"efficiency": eta, "cost_EUR_per_MWh": cost, "model_total_EUR": t.loc[M, "total_EUR"],
                     "model_sharpe": t.loc[M, "sharpe_annualised"],
                     "capture": t.loc[M, "capture_of_perfect_foresight"]})
sens = pd.DataFrame(rows)
sens.round(2).to_csv(C.RESULTS_DIR / "backtest_sensitivity.csv", index=False)
print("\nSensitivity (test period):\n", sens.round(2).to_string(index=False))
json.dump(extras, open(C.RESULTS_DIR / "backtest_extras.json", "w"), indent=1)

# stability over time (walk-forward out-of-sample predictions for the whole 2024-04..2026-09 span)
full_tab, full = table(P, cols)
fp = pd.DataFrame(full)
fp.index = pd.to_datetime(fp.index)
fp.round(2).to_csv(C.RESULTS_DIR / "backtest_daily_pnl_all.csv")
by_q = fp.groupby(fp.index.to_period("Q")).agg(["sum"]).droplevel(1, axis=1)
by_q["model_sharpe"] = fp.groupby(fp.index.to_period("Q"))[M].apply(lambda s: s.mean() / s.std() * np.sqrt(365))
by_q["capture"] = by_q[M] / by_q["perfect"]
by_q.round(2).to_csv(C.RESULTS_DIR / "backtest_by_quarter.csv")
print("\nBy quarter (walk-forward out-of-sample; 2025-Q2 to 2026-Q1 is the development window):\n", by_q.round(2).to_string())

# figures
fig, ax = plt.subplots(figsize=(10, 4.3))
cum = fp.cumsum()
for c, lab, kw in [("perfect", "perfect foresight (upper bound)", dict(color="grey", ls=":")),
                   (M, "model", dict(color="C0", lw=2)),
                   ("naive_yesterday", "naive: yesterday's prices", dict(color="C1")),
                   ("naive_lastweek", "naive: last week's prices", dict(color="C2"))]:
    ax.plot(cum.index, cum[c] / 1000, label=lab, **kw)
ax.axvspan(pd.Timestamp(C.DEV_START), pd.Timestamp(C.TEST_START), color="orange", alpha=0.08, label="development window")
ax.axvspan(pd.Timestamp(C.TEST_START), pd.Timestamp(C.TEST_END), color="green", alpha=0.10, label="final test")
ax.set_ylabel("cumulative profit, kEUR per MW (2 MWh battery)"); ax.legend(fontsize=7)
ax.set_title("Virtual battery, day-ahead market: cumulative profit"); fig.tight_layout()
fig.savefig(C.FIG_DIR / "backtest_cumulative_profit.png", dpi=130); plt.close(fig)

tm = fp.loc[C.TEST_START:, [M, "naive_yesterday", "perfect"]].groupby(lambda d: d.strftime("%Y-%m")).sum()
fig, ax = plt.subplots(figsize=(8, 4)); (tm / 1000).plot.bar(ax=ax, rot=0)
ax.set_ylabel("kEUR per month"); ax.set_title("Test period: monthly profit"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(C.FIG_DIR / "backtest_monthly_profit.png", dpi=130); plt.close(fig)

# one example day: median-profit day of the test
d_med = pnl[M].sub(pnl[M].median()).abs().idxmin()
g = test[test["local_date"] == d_med]
a = B.schedule(g[M].to_numpy(float), ETA, COST)
fig, ax = plt.subplots(figsize=(9, 4)); x = g.index.tz_convert("Europe/Berlin").hour
ax.plot(x, g["price"], "k", label="actual price"); ax.plot(x, g[M], "C0--", label="model forecast")
ax.scatter(x[a == 1], g["price"][a == 1], c="green", s=70, label="charge", zorder=3)
ax.scatter(x[a == -1], g["price"][a == -1], c="red", s=70, label="discharge", zorder=3)
ax.set_xlabel("hour (local)"); ax.set_ylabel("EUR/MWh"); ax.legend(fontsize=8)
ax.set_title(f"Example day {pd.Timestamp(d_med).date()} (median-profit day)"); fig.tight_layout()
fig.savefig(C.FIG_DIR / "backtest_example_day.png", dpi=130); plt.close(fig)
print("\nsaved results and figures")
