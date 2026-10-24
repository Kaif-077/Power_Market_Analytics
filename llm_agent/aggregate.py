"""Turn extracted records into a daily signal table per country (by the date an event starts)."""
import pandas as pd

THERMAL = {"gas", "coal", "lignite", "nuclear", "other"}


def build_signals(records: list) -> pd.DataFrame:
    rows = []
    for r in records:
        if r is None or not r.get("start_time") or r["event_type"] == "other":
            continue
        sign = {"up": 1, "down": -1, "neutral": 0}[r["direction"]]
        mw = r["capacity_mw"] or 0.0
        rows.append({"date": r["start_time"][:10], "zone": r["zone"] or "unknown",
                     "unplanned_thermal_mw": mw if (r["status"] == "unplanned" and r["fuel"] in THERMAL) else 0.0,
                     "planned_mw_offline": mw if r["status"] == "planned" else 0.0,
                     "grid_mw_out": mw if r["event_type"] == "grid_outage" else 0.0,
                     "net_direction_score": sign * r["confidence"], "n_events": 1})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby(["date", "zone"], as_index=False).sum()
