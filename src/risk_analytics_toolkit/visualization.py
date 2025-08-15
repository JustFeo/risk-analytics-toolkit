from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_returns_histogram(
    returns: np.ndarray, var: float | None = None, cvar: float | None = None
):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(returns, bins=50, alpha=0.7, color="steelblue")
    ax.set_title("Returns histogram")
    ax.set_xlabel("Return")
    ax.set_ylabel("Count")
    if var is not None:
        ax.axvline(-var, color="red", linestyle="--", label=f"VaR={var:.4f}")
    if cvar is not None:
        ax.axvline(-cvar, color="orange", linestyle=":", label=f"CVaR={cvar:.4f}")
    ax.legend(loc="best")
    return fig


def plot_simulated_paths(paths: np.ndarray, times: np.ndarray | None = None, n_plot: int = 50):
    fig, ax = plt.subplots(figsize=(7, 4))
    m = min(n_plot, paths.shape[0])
    X = times if times is not None else np.arange(paths.shape[1])
    for i in range(m):
        ax.plot(X, paths[i], alpha=0.6)
    ax.set_title("Simulated paths")
    return fig


def plot_drawdown(series: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(series, color="firebrick")
    ax.set_title("Drawdown series")
    return fig
