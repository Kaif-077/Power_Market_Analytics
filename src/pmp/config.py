"""Project-wide settings. Raw data location can be overridden with PMP_RAW_DIR."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("PMP_RAW_DIR", ROOT / "data" / "raw"))
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Weather day-ahead forecasts are complete at all 8 locations from 2024-02-16 (Lower Saxony
# starts 3 h later), so modelling starts on the first full day after that.
DATA_START = "2024-02-17 00:00"
DATA_END = "2026-09-30 23:00"

# Development period: walk-forward folds used to choose model settings.
DEV_START = "2025-04-01"
# Final, untouched test period (never used for tuning).
TEST_START = "2026-04-01"
TEST_END = "2026-10-01"  # exclusive

# Days removed between the end of training data and the start of a test month.
EMBARGO_DAYS = 2

WEATHER_VARS = [
    "wind_speed_100m", "wind_direction_100m", "wind_gusts_10m",
    "temperature_2m", "surface_pressure",
]
ONSHORE_LOCS = ["sh_west_coast", "lower_saxony", "brandenburg", "saxony_anhalt", "nrw_hesse"]
OFFSHORE_LOCS = ["north_sea_offshore", "baltic_offshore"]
WIND_LOCS = ONSHORE_LOCS + OFFSHORE_LOCS
ALL_LOCS = WIND_LOCS + ["bavaria_south"]
