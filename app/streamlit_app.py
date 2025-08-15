
import numpy as np
import pandas as pd
import streamlit as st

from risk_analytics_toolkit.data.ingestion import fetch_yfinance
from risk_analytics_toolkit.metrics import cvar_historical, var_historical
from risk_analytics_toolkit.simulations import simulate_gbm
from risk_analytics_toolkit.visualization import (
    plot_returns_histogram,
    plot_simulated_paths,
)

st.set_page_config(page_title="Risk Analytics Toolkit", layout="wide")
st.title("Risk Analytics Toolkit (student edition)")

with st.sidebar:
    st.header("Data")
    source = st.selectbox("Source", ["yfinance", "CSV upload"])
    alpha = st.slider("alpha (VaR/CVaR)", 0.001, 0.2, 0.05)
    n_sim = st.slider("# GBM paths", 100, 5000, 1000, step=100)

if source == "yfinance":
    ticker = st.text_input("Ticker", "SPY")
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
    if st.button("Load data"):
        data = fetch_yfinance(ticker, period)
        st.session_state["data"] = data
else:
    upload = st.file_uploader("Upload CSV (needs date & price cols)")
    date_col = st.text_input("date col name", "Date")
    price_col = st.text_input("price col name", "Close")
    if upload is not None:
        data = pd.read_csv(upload)
        if date_col in data.columns and price_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col])
            data.sort_values(by=date_col, inplace=True)
            data["Returns"] = data[price_col].pct_change().fillna(0.0)
            st.session_state["data"] = data.rename(columns={date_col: "Date", price_col: "Close"})[
                ["Date", "Close", "Returns"]
            ]

if "data" in st.session_state:
    df = st.session_state["data"]
    st.subheader("Summary")
    st.dataframe(df.tail(10), use_container_width=True)
    rets = df["Returns"].to_numpy()
    vh = var_historical(rets, alpha=alpha)
    ch = cvar_historical(rets, alpha=alpha)
    st.metric("VaR (hist)", f"{vh:.5f}")
    st.metric("CVaR (hist)", f"{ch:.5f}")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_returns_histogram(rets, var=vh, cvar=ch))
    with col2:
        mu = float(np.mean(rets))
        sigma = float(np.std(rets, ddof=1))
        paths = simulate_gbm(
            S0=float(df["Close"].iloc[-1]), mu=mu, sigma=sigma, dt=1 / 252, steps=252, n_paths=n_sim
        )
        st.pyplot(plot_simulated_paths(paths, n_plot=50))

    st.download_button("Export returns", df.to_csv(index=False), file_name="returns.csv")
