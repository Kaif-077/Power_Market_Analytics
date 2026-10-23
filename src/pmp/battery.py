"""Virtual battery trading in the day-ahead market, and performance measures.

Battery: 1 MW power, 2 MWh energy, at most one full cycle (2 MWh discharged) per day, empty at
the start and end of every delivery day. Each hour it charges 1 MWh, discharges 1 MWh or idles.
Charging 1 MWh costs price + c; discharging earns eta * (price - c), where eta is the round-trip
efficiency and c is a cost per MWh traded (fees, balancing, wear). Decisions are made from
PREDICTED prices; profit is booked at ACTUAL prices.
"""
import numpy as np
import pandas as pd

NEG = -1e18


def schedule(pred, eta=0.88, cost=2.0, soc_max=2, daily_discharge_cap=2):
    """Best actions (+1 charge, -1 discharge, 0 idle) for predicted prices of one day (dynamic programming)."""
    T = len(pred)
    V = np.full((T + 1, soc_max + 1, daily_discharge_cap + 1), NEG)
    V[T, 0, :] = 0.0  # must end the day empty
    act = np.zeros((T, soc_max + 1, daily_discharge_cap + 1), dtype=np.int8)
    for t in range(T - 1, -1, -1):
        p = pred[t]
        for s in range(soc_max + 1):
            for u in range(daily_discharge_cap + 1):
                best, a = V[t + 1, s, u], 0
                if s < soc_max:
                    v = -(p + cost) + V[t + 1, s + 1, u]
                    if v > best:
                        best, a = v, 1
                if s > 0 and u < daily_discharge_cap:
                    v = eta * (p - cost) + V[t + 1, s - 1, u + 1]
                    if v > best:
                        best, a = v, -1
                V[t, s, u], act[t, s, u] = best, a
    out = np.zeros(T, dtype=np.int8)
    s = u = 0
    for t in range(T):
        a = act[t, s, u]
        out[t] = a
        if a == 1:
            s += 1
        elif a == -1:
            s -= 1
            u += 1
    return out


def profit(actions, actual, eta=0.88, cost=2.0):
    actual = np.asarray(actual, dtype=float)
    ch, dis = actions == 1, actions == -1
    return float(-(actual[ch] + cost).sum() + eta * (actual[dis] - cost).sum())


def daily_pnl(df, pred_col, eta=0.88, cost=2.0, price_col="price"):
    """Daily profit (EUR) per delivery day. Days with fewer than 23 hours (partial days) are skipped."""
    rows = {}
    for day, g in df.groupby("local_date"):
        if len(g) < 23 or g[pred_col].isna().any() or g[price_col].isna().any():
            continue
        a = schedule(g[pred_col].to_numpy(float), eta, cost)
        rows[day] = profit(a, g[price_col].to_numpy(float), eta, cost)
    return pd.Series(rows, name=pred_col)


def perf(pnl, perfect=None):
    cum = pnl.cumsum()
    dd = (cum - cum.cummax()).min()
    sd = pnl.std()
    out = {
        "days": int(len(pnl)),
        "total_EUR": float(pnl.sum()),
        "mean_EUR_per_day": float(pnl.mean()),
        "std_EUR_per_day": float(sd),
        "sharpe_annualised": float(pnl.mean() / sd * np.sqrt(365)) if sd > 0 else np.nan,
        "max_drawdown_EUR": float(dd),
        "worst_day_EUR": float(pnl.min()),
        "share_days_profitable": float((pnl > 0).mean()),
    }
    if perfect is not None:
        out["capture_of_perfect_foresight"] = float(pnl.sum() / perfect.reindex(pnl.index).sum())
    return out


def bootstrap(pnl_a, pnl_b=None, n_boot=4000, seed=0):
    """95% interval for mean daily profit (or for the mean difference a - b) and for the Sharpe ratio of a."""
    rng = np.random.default_rng(seed)
    d = (pnl_a - pnl_b).dropna().to_numpy() if pnl_b is not None else pnl_a.dropna().to_numpy()
    m = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    out = {"mean": float(d.mean()), "ci95_low": float(np.percentile(m, 2.5)), "ci95_high": float(np.percentile(m, 97.5))}
    if pnl_b is None:
        x = rng.choice(d, size=(n_boot, len(d)), replace=True)
        sh = x.mean(axis=1) / x.std(axis=1, ddof=1) * np.sqrt(365)
        out["sharpe_ci95_low"], out["sharpe_ci95_high"] = float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))
    return out
