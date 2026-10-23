"""Features for predicting day-ahead prices of delivery day D.

Decision time is 12:00 CET on D-1 (auction close). Known at that time:
- forecasts of load, wind and solar for day D (ENTSO-E day-ahead forecasts)
- all day-ahead prices up to and including day D-1 (cleared on D-2)
- gas and carbon closes dated D-2 or earlier (see data.fuel_prices_for_delivery_days)
"""
import numpy as np
import pandas as pd

from . import data

LOCAL_TZ = "Europe/Berlin"

FEATURES = [
    "hour_local", "dow", "doy_sin", "doy_cos",
    "load_fc", "solar_fc", "wind_on_fc", "wind_off_fc", "wind_fc",
    "resid_load", "rl_day_mean", "rl_rel", "rl_rank", "rl_diff1", "rl_diff3",
    "ttf", "eua", "fuel_cost",
    "p_lag24", "p_lag48", "p_lag168",
    "p_yday_mean", "p_yday_max", "p_yday_min", "p_yday_std", "p_7d_mean",
]


def build_price_dataset(prices: pd.Series, fund: pd.DataFrame, fuel: pd.DataFrame) -> pd.DataFrame:
    idx = prices.index
    df = pd.DataFrame({"price": prices}, index=idx)
    loc = idx.tz_convert(LOCAL_TZ)
    df["local_date"] = loc.tz_localize(None).normalize()
    df["hour_local"] = loc.hour
    df["dow"] = loc.dayofweek
    doy = loc.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    f = fund.reindex(idx)
    for c in ("load_fc", "solar_fc", "wind_on_fc", "wind_off_fc"):
        df[c] = f[c]
    df["wind_fc"] = df["wind_on_fc"] + df["wind_off_fc"]
    df["resid_load"] = df["load_fc"] - df["wind_fc"] - df["solar_fc"]
    g = df.groupby("local_date")["resid_load"]
    df["rl_day_mean"] = g.transform("mean")
    df["rl_rel"] = df["resid_load"] - df["rl_day_mean"]
    df["rl_rank"] = g.rank(pct=True)
    df["rl_diff1"] = df["resid_load"].diff(1)
    df["rl_diff3"] = df["resid_load"].diff(3)

    days = pd.DatetimeIndex(sorted(df["local_date"].unique()))
    fu = data.fuel_prices_for_delivery_days(fuel, days)
    df["ttf"] = df["local_date"].map(fu["ttf"])
    df["eua"] = df["local_date"].map(fu["eua"])
    # rough marginal cost of a gas plant (EUR/MWh electric): 55% efficiency, 0.202 tCO2/MWh gas
    df["fuel_cost"] = df["ttf"] / 0.55 + 0.202 / 0.55 * df["eua"]

    # Lagged prices. A 24 h lag on the last hour of a 25-hour day would point at the SAME delivery
    # day (not yet known), so lags that land on the same local day are blanked.
    lag24 = prices.shift(24)
    same_day = df["local_date"].shift(24) >= df["local_date"]
    df["p_lag24"] = lag24.where(~same_day)
    df["p_lag48"] = prices.shift(48)
    df["p_lag168"] = prices.shift(168)
    daily = df.groupby("local_date")["price"].agg(["mean", "max", "min", "std"])
    prev = daily.shift(1)  # previous calendar day
    for c in ("mean", "max", "min", "std"):
        df[f"p_yday_{c}"] = df["local_date"].map(prev[c])
    df["p_7d_mean"] = df["local_date"].map(daily["mean"].rolling(7).mean().shift(1))
    df["n_hours_in_day"] = df.groupby("local_date")["price"].transform("size")
    return df
