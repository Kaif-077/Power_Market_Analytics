"""Scramble prices from the start of local day X onward; every feature of day X must be unchanged
(day X may only use prices up to day X-1). Run with:  python tests/test_price_no_leakage.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pmp import data, price_features as PF

prices, fund, fuel = data.load_da_prices(), data.load_fundamental_forecasts(), data.load_fuel_prices()
base = PF.build_price_dataset(prices, fund, fuel)

for X in ("2025-03-30", "2025-10-26", "2025-06-10"):  # includes both clock-change days
    X = pd.Timestamp(X)
    start_utc = base.index[base["local_date"] >= X][0]
    scr = prices.copy()
    scr.loc[scr.index >= start_utc] = np.random.default_rng(1).uniform(-100, 400, (scr.index >= start_utc).sum())
    alt = PF.build_price_dataset(scr, fund, fuel)
    cols = [c for c in PF.FEATURES]
    day = base["local_date"] == X
    pd.testing.assert_frame_equal(base.loc[day, cols], alt.loc[day, cols])
    # and rows before day X are of course unchanged too
    before = base["local_date"] < X
    pd.testing.assert_frame_equal(base.loc[before, cols], alt.loc[before, cols])
    after = base["local_date"] == X + pd.Timedelta(days=1)
    assert (base.loc[after, cols].fillna(-9).to_numpy() != alt.loc[after, cols].fillna(-9).to_numpy()).any()
print("OK: day-D features use no price from day D or later (incl. 23/25-hour days)")
