"""Walk-forward validation, metrics and a day-block bootstrap."""
import numpy as np
import pandas as pd

from . import config as C
from .models import PowerCurveBaseline, make_gbm


def month_folds(start, end):
    """List of (month_start, next_month_start) timestamps in UTC, end exclusive."""
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    folds = []
    while s < e:
        n = s + pd.DateOffset(months=1)
        folds.append((s, min(n, e)))
        s = n
    return folds


def _train_mask(index, month_start):
    # Train only on hours that end before the test month starts, minus an embargo gap.
    return index < month_start - pd.Timedelta(days=C.EMBARGO_DAYS)


def walk_forward_gbm(df, feat_cols, folds, normalize=True, loss="squared_error", quantile=None,
                     half_life_days=None, base_col=None, **gbm_kw):
    """Expanding-window walk-forward: for every month, retrain on all earlier data and
    predict that month. Returns predictions in MW for all test hours."""
    cap = df["cap_proxy"]
    if base_col:  # residual mode: learn the correction to an existing forecast
        target = df["y"] - df[base_col]
    else:
        target = df["y"] / cap if normalize else df["y"]
    preds = []
    for ms, me in folds:
        tr = _train_mask(df.index, ms) & target.notna()
        te = (df.index >= ms) & (df.index < me)
        model = make_gbm(loss=loss, quantile=quantile, **gbm_kw)
        w = None
        if half_life_days:  # recent data counts more (installed capacity keeps growing)
            age = (ms - df.index[tr]).total_seconds() / 86400.0
            w = np.power(0.5, age / half_life_days)
        model.fit(df.loc[tr, feat_cols], target[tr], sample_weight=w)
        p = model.predict(df.loc[te, feat_cols])
        if base_col:
            p = p + df.loc[te, base_col].to_numpy()
        elif normalize:
            p = p * cap[te].to_numpy()
        preds.append(pd.Series(p, index=df.index[te]))
    return pd.concat(preds)


def walk_forward_power_curve(df, folds):
    cap = df["cap_proxy"]
    cf = df["y"] / cap
    preds = []
    for ms, me in folds:
        tr = _train_mask(df.index, ms)
        te = (df.index >= ms) & (df.index < me)
        pc = PowerCurveBaseline().fit(df.loc[tr, "ws_mean_all"], cf[tr])
        p = pc.predict(df.loc[te, "ws_mean_all"]) * cap[te].to_numpy()
        preds.append(pd.Series(p, index=df.index[te]))
    return pd.concat(preds)


def persistence_48h(df, index):
    """Naive baseline: generation at the same hour two days earlier (latest data allowed)."""
    return df["gen_lag48"].reindex(index)


def metrics(y, p):
    m = y.notna() & p.notna()
    e = (p[m] - y[m])
    return {
        "n": int(m.sum()),
        "MAE": float(e.abs().mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "bias": float(e.mean()),
        "MAE_pct_of_mean": float(e.abs().mean() / y[m].mean() * 100),
    }


def day_block_bootstrap(y, p_model, p_ref, n_boot=4000, seed=0):
    """Bootstrap over whole days: difference in MAE (model - reference). Negative = model better."""
    m = y.notna() & p_model.notna() & p_ref.notna()
    d = pd.DataFrame({
        "model": (p_model[m] - y[m]).abs(),
        "ref": (p_ref[m] - y[m]).abs(),
    })
    daily = d.groupby(d.index.floor("D")).mean()
    diff = (daily["model"] - daily["ref"]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return {
        "days": int(len(diff)),
        "mean_MAE_diff": float(diff.mean()),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "share_days_model_better": float((diff < 0).mean()),
    }
