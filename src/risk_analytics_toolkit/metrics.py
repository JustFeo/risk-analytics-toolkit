from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


def compute_returns(price_series: np.ndarray) -> np.ndarray:
    """Compute simple returns from price series as r_t = P_t / P_{t-1} - 1.
    Assumes price_series is 1D and positive.
    """
    price_series = np.asarray(price_series, dtype=float)
    rets = np.zeros_like(price_series)
    rets[1:] = price_series[1:] / price_series[:-1] - 1.0
    return rets


def var_historical(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical VaR at level alpha (loss quantile).
    Convention: returns are arithmetic returns; loss = -returns.
    VaR_alpha = quantile of loss at alpha.
    """
    losses = -np.asarray(returns, dtype=float)
    return float(np.quantile(losses, alpha))


def cvar_historical(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical CVaR (Expected Shortfall) at level alpha.
    Mean of losses below VaR.
    """
    losses = -np.asarray(returns, dtype=float)
    var = np.quantile(losses, alpha)
    mask = losses >= var - 1e-12
    return float(losses[mask].mean())


def var_parametric_normal(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Parametric normal VaR: mu + sigma * z_alpha for losses (negative returns).
    Equivalent to - (mu + sigma * z_{1-alpha}) if using returns convention.
    Here we return loss quantile: VaR = -(mu + sigma * z_{alpha}) on returns.
    """
    r = np.asarray(returns, dtype=float)
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1) if r.size > 1 else 0.0)
    z = stats.norm.ppf(alpha)
    return float(-(mu + sigma * z))


def var_parametric_t(returns: np.ndarray, alpha: float = 0.05, nu: float = 5) -> float:
    """Student-t VaR using fixed nu degrees of freedom.
    VaR = -(mu + sigma * t_ppf(alpha, nu)).
    """
    r = np.asarray(returns, dtype=float)
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1) if r.size > 1 else 0.0)
    tq = stats.t.ppf(alpha, df=nu)
    return float(-(mu + sigma * tq))


def rolling_var(returns: np.ndarray, window: int, alpha: float) -> np.ndarray:
    """Rolling historical VaR over a moving window (naive loop, ok for demo)."""
    r = np.asarray(returns, dtype=float)
    out = np.full_like(r, np.nan)
    for i in range(window, len(r)):
        out[i] = var_historical(r[i - window : i], alpha)
    return out


def max_drawdown(price_series: np.ndarray) -> Tuple[float, int]:
    """Compute max drawdown and duration.
    Drawdown = (peak - price)/peak. Duration is longest consecutive days under peak.
    """
    p = np.asarray(price_series, dtype=float)
    peaks = np.maximum.accumulate(p)
    dd = (peaks - p) / peaks
    max_dd = float(np.max(dd))
    # duration: longest run where dd>0 until a new peak
    longest = current = 0
    for i in range(1, len(p)):
        if p[i] < peaks[i]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return max_dd, int(longest)
