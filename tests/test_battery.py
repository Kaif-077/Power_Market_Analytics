"""Hand-checkable cases for the battery scheduler. Run with:  python tests/test_battery.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from pmp import battery as B

# Charge at 10, discharge at 50, no losses: 2 MWh -> 2 * (50 - 10) = 80
p = np.array([10, 10, 50, 50.0]); a = B.schedule(p, 1.0, 0.0)
assert list(a) == [1, 1, -1, -1] and B.profit(a, p, 1.0, 0.0) == 80.0
# Never discharges more than 2 MWh per day, never charges when it cannot sell later
p = np.array([50, 10, 50, 10, 50, 10, 50.0]); a = B.schedule(p, 1.0, 0.0)
assert (a == -1).sum() == 2 and (a == 1).sum() == 2 and B.profit(a, p, 1.0, 0.0) == 80.0
# A small spread is eaten by efficiency loss and costs: stay idle
p = np.array([10, 12, 11, 13.0]); assert not B.schedule(p, 0.88, 2.0).any()
# Negative prices are worth charging at
p = np.array([-20, -20, 40, 40.0]); a = B.schedule(p, 1.0, 0.0); assert B.profit(a, p, 1.0, 0.0) == 120.0
# Always ends the day empty
rng = np.random.default_rng(0)
for _ in range(200):
    p = rng.uniform(-50, 200, 24); a = B.schedule(p)
    assert a.sum() == 0 and np.cumsum(a).min() >= 0 and np.cumsum(a).max() <= 2
print("OK: battery scheduler behaves as expected")
