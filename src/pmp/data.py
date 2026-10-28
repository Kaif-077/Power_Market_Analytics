"""Loading and cleaning of the raw files. Everything is returned hourly, in UTC."""
import pandas as pd

from . import config as C


def _read_entsoe(name: str) -> pd.DataFrame:
    df = pd.read_csv(C.RAW_DIR / "entsoe" / f"{name}.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "time_utc"
    return df.sort_index()


def _merge_split_columns(df: pd.DataFrame) -> pd.Series:
    """entsoe-py changed its output format in July 2025, so the actual-generation files hold
    the data in different columns before and after that date. 'Consumption' columns are
    pump/own-use and are always zero. Keep the rest and fill one from the other."""
    cols = [c for c in df.columns if "Consumption" not in c]
    out = df[cols[0]].copy()
    for c in cols[1:]:
        out = out.combine_first(df[c])
    return out


def load_actuals() -> pd.DataFrame:
    """Actual German wind generation in MW (hourly average)."""
    on = _merge_split_columns(_read_entsoe("actual_wind_onshore"))
    off = _merge_split_columns(_read_entsoe("actual_wind_offshore"))
    out = pd.DataFrame({"onshore": on, "offshore": off})
    out["wind_total"] = out["onshore"] + out["offshore"]
    return out


def load_entsoe_forecast() -> pd.DataFrame:
    """ENTSO-E's own day-ahead wind forecast (the benchmark). The 3 missing hours per year
    fall on the clock-change night and are filled forward."""
    f = _read_entsoe("forecast_wind_solar_dayahead")
    f = f[["Wind Onshore", "Wind Offshore"]].ffill(limit=3)
    f["entsoe_fc"] = f["Wind Onshore"] + f["Wind Offshore"]
    return f[["entsoe_fc"]]


def load_weather() -> pd.DataFrame:
    """Long-format day-ahead weather forecasts (Open-Meteo Previous Runs API, day 1)."""
    df = pd.read_csv(C.RAW_DIR / "weather" / "weather_previous_runs_day1.csv")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df


def load_fuel_prices() -> pd.DataFrame:
    """Daily closing prices: ttf (EUR/MWh, ICE Dutch TTF front month) and eua (EUR/t CO2,
    EU carbon futures). Source files are Investing.com exports (US date format).
    TTF has low-volume Sunday evening sessions from 2026; they are dropped so that both
    series share the same Monday-Friday calendar."""
    files = {
        "ttf": "ICE_Dutch_TTF_Natural_Gas_Futures_Historical_Data.csv",
        "eua": "Carbon_Emissions_Futures_Historical_Data.csv",
    }
    out = {}
    for key, fname in files.items():
        d = pd.read_csv(C.RAW_DIR / "fuels" / fname, encoding="utf-8-sig")
        d["Date"] = pd.to_datetime(d["Date"], format="%m/%d/%Y")
        s = pd.Series(pd.to_numeric(d["Price"], errors="coerce").to_numpy(), index=d["Date"]).sort_index()
        out[key] = s[s.index.dayofweek < 5]
    return pd.DataFrame(out)


def fuel_prices_for_delivery_days(fuel: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Timing rule (no look-ahead): the day-ahead auction for delivery day D closes at 12:00 CET
    on D-1, before D-1's closing price exists. So delivery day D may only use the latest close
    dated D-2 or earlier (weekends and holidays carry the last close forward)."""
    cal = fuel.reindex(pd.date_range(fuel.index.min(), max(days.max(), fuel.index.max()), freq="D")).ffill()
    out = cal.reindex(days - pd.Timedelta(days=2))
    out.index = days
    return out


def _smard_number(s: pd.Series) -> pd.Series:
    """SMARD uses German number format: '4.378,50' means 4378.50."""
    return pd.to_numeric(s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")


def load_da_prices() -> pd.Series:
    """German-Luxembourg day-ahead price in EUR/MWh, hourly, UTC index. SMARD timestamps are
    German local time (23- and 25-hour days at the clock changes), converted here to UTC.
    Since Oct 2025 the market trades in 15-minute blocks; SMARD gives the hourly average."""
    f = next((C.RAW_DIR / "smard").glob("Gro_handelspreise*.csv"))
    d = pd.read_csv(f, sep=";", encoding="utf-8-sig", dtype=str)
    col = [c for c in d.columns if c.startswith("Deutschland/Luxemburg")][0]
    start = pd.to_datetime(d["Datum von"], format="%d.%m.%Y %H:%M")
    idx = start.dt.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward").dt.tz_convert("UTC")
    return pd.Series(_smard_number(d[col]).to_numpy(), index=pd.DatetimeIndex(idx, name="time_utc"), name="price")


def load_fundamental_forecasts() -> pd.DataFrame:
    """ENTSO-E day-ahead forecasts of load, wind and solar (MW), hourly UTC."""
    f = _read_entsoe("forecast_wind_solar_dayahead")[["Solar", "Wind Onshore", "Wind Offshore"]]
    l = _read_entsoe("forecast_load_dayahead")[["Forecasted Load"]]
    out = f.join(l).ffill(limit=3)
    out.columns = ["solar_fc", "wind_on_fc", "wind_off_fc", "load_fc"]
    return out
