"""If we scramble actual generation after time T, every feature row whose forecast was made
before T must stay exactly the same. Run with:  python tests/test_no_leakage.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pmp import data, features as F

weather, actuals, fc = data.load_weather(), data.load_actuals(), data.load_entsoe_forecast()
base = F.build_dataset(weather, actuals, fc)

T = pd.Timestamp("2025-06-15 12:00", tz="UTC")
scrambled = actuals.copy()
rng = np.random.default_rng(1)
later = scrambled.index > T
scrambled.loc[later, "wind_total"] = rng.uniform(0, 60000, later.sum())
alt = F.build_dataset(weather, scrambled, fc)

obs_cols = ["cap_proxy", "gen_cut_last", "gen_cut_mean24", "gen_lag48"]
# Rows of day D may use actuals up to D-2 23:00, i.e. day_start - 25h <= T must be unaffected.
safe = (base.index.floor("D") - F.CUTOFF_LAG) <= T
assert safe.sum() > 1000
pd.testing.assert_frame_equal(base.loc[safe, obs_cols], alt.loc[safe, obs_cols])
changed = ~safe & (base[obs_cols].to_numpy() != alt[obs_cols].to_numpy()).any(axis=1)
assert changed.sum() > 0, "test is too weak: scrambling should change rows after the cutoff"
print(f"OK: {safe.sum()} rows unaffected by future actuals; {changed.sum()} later rows do change (as expected).")
