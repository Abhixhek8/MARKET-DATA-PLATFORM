"""Streamlit dashboard. Run: streamlit run dashboard/app.py"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mdp import analytics, ml
from mdp.config import get_settings
from mdp.warehouse import Warehouse

st.set_page_config(page_title="Market Data Platform", layout="wide")
st.title("Market Data Platform")


@st.cache_data(ttl=300)
def load_prices() -> pd.DataFrame:
    wh = Warehouse(get_settings().db_path)
    try:
        return wh.prices()
    finally:
        wh.close()


data = load_prices()
if data.empty:
    st.warning("Warehouse is empty. Run `python -m mdp.pipeline run --source synthetic` first.")
    st.stop()

tickers = sorted(data["ticker"].unique())
ticker = st.sidebar.selectbox("Ticker", tickers)
horizon = st.sidebar.slider("Forecast horizon (days)", 1, 30, 5)
z = st.sidebar.slider("Anomaly z-score threshold", 2.0, 5.0, 3.0, 0.5)

one = data[data["ticker"] == ticker]
m = analytics.add_metrics(one)
s = analytics.summary(one)

c = st.columns(5)
c[0].metric("Last close", f"{s['last_close']:.2f}")
c[1].metric("Total return", f"{s['total_return']:.1%}")
c[2].metric("Volatility (ann.)", f"{s['annualised_volatility']:.1%}")
c[3].metric("Sharpe", s["sharpe_ratio"])
c[4].metric("Max drawdown", f"{s['max_drawdown']:.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["Price & trend", "Market overview", "Forecast", "Anomalies"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=m["date"], open=m["open"], high=m["high"],
                                 low=m["low"], close=m["close"], name="OHLC"))
    fig.add_trace(go.Scatter(x=m["date"], y=m["sma_20"], name="SMA 20"))
    fig.add_trace(go.Scatter(x=m["date"], y=m["sma_50"], name="SMA 50"))
    fig.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.line(m, x="date", y="volatility_20d", title="20-day annualised volatility"),
                    use_container_width=True)

wide = data.pivot(index="date", columns="ticker", values="close")
with tab2:
    left, right = st.columns(2)
    movers = analytics.top_movers(wide)
    left.subheader("Top gainers (1d)")
    left.dataframe(movers["gainers"].map("{:.2%}".format).rename("return"))
    right.subheader("Top losers (1d)")
    right.dataframe(movers["losers"].map("{:.2%}".format).rename("return"))
    st.plotly_chart(px.imshow(analytics.return_correlation(wide), zmin=-1, zmax=1,
                              color_continuous_scale="RdBu_r", title="Return correlation"),
                    use_container_width=True)
    st.plotly_chart(px.imshow(analytics.volatility_heatmap(wide), aspect="auto",
                              color_continuous_scale="YlOrRd", title="Monthly volatility heatmap"),
                    use_container_width=True)

with tab3:
    series = one.set_index("date")["close"]
    fc = ml.forecast(series, horizon)
    fdf = pd.DataFrame(fc["forecast"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index[-90:], y=series.iloc[-90:], name="History"))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["predicted_close"], name="Forecast"))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["upper"], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["lower"], fill="tonexty", line=dict(width=0),
                             name="95% band"))
    st.plotly_chart(fig, use_container_width=True)
    st.json(fc["backtest"])
    st.caption("Backtest compares the model against a naive 'price stays flat' baseline. "
               "Short-horizon price forecasting is hard; treat this as a demo, not advice.")

with tab4:
    an = ml.detect_anomalies(one.set_index("date")["close"], z_threshold=z)
    st.write(f"{len(an)} anomalous days at |z| > {z}")
    st.dataframe(an)
