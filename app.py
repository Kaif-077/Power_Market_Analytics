"""Dashboard for the power market project. Reads only the files in results/ (no raw data needed).
Run locally:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

R = Path(__file__).parent / "results"
st.set_page_config(page_title="Power Market Analytics", layout="wide")


@st.cache_data
def read(name, **kw):
    return pd.read_csv(R / name, **kw)


st.title("European Power Market Analytics")
st.caption("German day-ahead wind forecasting and a virtual-battery back-test. "
           "Final test period: April to September 2026, never used for tuning.")

tab1, tab2, tab3 = st.tabs(["Wind forecast", "Battery back-test", "Method and limits"])

with tab1:
    m = read("metrics_test.csv", index_col=0)
    names = {"entsoe_benchmark": "ENTSO-E day-ahead (benchmark)", "hybrid_entsoe_correction": "Hybrid: ENTSO-E + correction",
             "gbm_weather_plus_obs": "Weather-based model", "power_curve": "Simple power curve",
             "persistence_48h": "Persistence (2 days ago)"}
    show = m.rename(index=names)[["MAE", "RMSE", "bias", "MAE_vs_entsoe_pct"]]
    show.columns = ["MAE (MW)", "RMSE (MW)", "Bias (MW)", "MAE vs ENTSO-E (%)"]
    st.subheader("Forecast error on the test period")
    st.dataframe(show.loc[[names[k] for k in names if k in m.index]].round(1), use_container_width=True)

    p = read("predictions_test.csv", index_col=0, parse_dates=True)
    days = sorted(set(p.index.date))
    a, b = st.select_slider("Date range", options=days, value=(days[0], days[13]))
    sl = p.loc[str(a):str(b)]
    st.line_chart(sl[["actual", "entsoe_benchmark", "hybrid_entsoe_correction", "gbm_weather_plus_obs"]]
                  .rename(columns={"entsoe_benchmark": "ENTSO-E", "hybrid_entsoe_correction": "Hybrid",
                                   "gbm_weather_plus_obs": "Weather model"}))
    st.subheader("Average error by month (MW)")
    st.line_chart(read("test_mae_by_month.csv", index_col=0).rename(columns={
        "entsoe_benchmark": "ENTSO-E", "gbm_weather_plus_obs": "Weather model", "hybrid_entsoe_correction": "Hybrid"}))

with tab2:
    t = read("backtest_test.csv", index_col=0)
    lab = {"perfect": "Perfect foresight (upper limit)", "model_level": "Model", "naive_yesterday": "Naive: yesterday's prices",
           "naive_lastweek": "Naive: last week's prices"}
    tt = t.rename(index=lab)[["days", "total_EUR", "mean_EUR_per_day", "capture_of_perfect_foresight", "max_drawdown_EUR"]]
    tt.columns = ["Days", "Total profit (EUR)", "EUR per day", "Share of perfect foresight", "Max drawdown (EUR)"]
    st.subheader("Virtual 1 MW / 2 MWh battery, final test")
    st.dataframe(tt.round(2), use_container_width=True)
    st.caption("Every day was profitable, so the Sharpe ratio is not a useful risk measure here.")

    pnl = read("backtest_daily_pnl_all.csv", index_col=0, parse_dates=True).cumsum() / 1000
    st.subheader("Cumulative profit (kEUR), Apr 2024 to Sep 2026")
    st.line_chart(pnl.rename(columns=lab))

    s = read("backtest_sensitivity.csv")
    st.subheader("Sensitivity of test-period profit (EUR) to the assumptions")
    st.dataframe(s.pivot(index="efficiency", columns="cost_EUR_per_MWh", values="model_total_EUR").round(0),
                 use_container_width=True)

    pr = read("price_predictions.csv", index_col=0, parse_dates=True)
    d = st.date_input("Look at one day's prices", value=pd.Timestamp("2026-06-15"),
                      min_value=pr.index.min().date(), max_value=pr.index.max().date())
    day = pr[pr["local_date"] == str(d)]
    if len(day) >= 23:
        day = day.reset_index(drop=True)
        day.index = [f"{i:02d}" for i in day.index]  # hour of the day, in order
        st.line_chart(day[["price", "model_level", "naive_yesterday"]].rename(
            columns={"price": "Actual", "model_level": "Model", "naive_yesterday": "Yesterday"}))
    else:
        st.info("No complete day of data for that date.")

with tab3:
    st.markdown("""
**Method.** Walk-forward validation: each month the model is retrained on earlier data only. Settings were chosen on
April 2025 to March 2026; April to September 2026 is the final test. Features use only information available before the
day-ahead auction, and automated tests check for leakage.

**Limits.**
- The weather-only wind model is worse than ENTSO-E; only the hybrid beats it, by a small margin.
- ENTSO-E's forecast may be published after the auction closes, so the hybrid may not be usable live.
- Most battery profit comes from the daily price shape, which naive forecasts also capture.
- SMARD gives hourly averages, while the market now trades in 15-minute blocks.
- 2026 price spreads were unusually wide, so profit per day should not be extrapolated.
- Not modelled: investment cost, grid fees, market impact, battery wear beyond a flat cost.
""")
