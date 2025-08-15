from __future__ import annotations

from typing import Dict

import numpy as np

from .data.ingestion import fetch_yfinance
from .metrics import cvar_historical, var_historical
from .simulations import simulate_gbm
from .utils import fig_to_base64
from .visualization import plot_returns_histogram, plot_simulated_paths


def analyze_ticker(ticker: str, period: str = "2y", alpha: float = 0.05, n_sim: int = 1000) -> Dict:
    """Quick analysis pipeline for a ticker (public data only).
    - fetch prices & returns
    - compute historical VaR/CVaR
    - simulate GBM as a toy model
    - return figures encoded for transport
    """
    df = fetch_yfinance(ticker, period=period)
    rets = df["Returns"].to_numpy()
    vh = var_historical(rets, alpha=alpha)
    ch = cvar_historical(rets, alpha=alpha)
    mu = float(np.mean(rets))
    sigma = float(np.std(rets, ddof=1))
    paths = simulate_gbm(
        S0=float(df["Close"].iloc[-1]), mu=mu, sigma=sigma, dt=1 / 252, steps=252, n_paths=n_sim
    )
    fig_hist = plot_returns_histogram(rets, var=vh, cvar=ch)
    fig_paths = plot_simulated_paths(paths, n_plot=30)
    return {
        "var_historical": float(vh),
        "cvar_historical": float(ch),
        "sim_ruin_prob": 0.0,  # not applicable in this quick pipeline
        "lundberg_bound": 0.0,
        "figures": {
            "hist": fig_to_base64(fig_hist),
            "paths": fig_to_base64(fig_paths),
        },
    }
