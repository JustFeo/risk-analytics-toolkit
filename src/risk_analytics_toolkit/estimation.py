from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import stats


def fit_gbm_params(returns: np.ndarray) -> Tuple[float, float]:
    """MLE for GBM drift and volatility using log returns.
    Assumes returns are arithmetic; approximate log returns by log(1+r).
    mu_hat = mean(log(1+r)) / dt (taking dt=1 unit); sigma_hat = std(log(1+r)).
    """
    r = np.asarray(returns, dtype=float)
    lr = np.log1p(r)
    mu = float(np.mean(lr))
    sigma = float(np.std(lr, ddof=1) if lr.size > 1 else 0.0)
    return mu, sigma


def fit_poisson_intensity(arrival_times: np.ndarray) -> float:
    """Estimate Poisson rate lambda via MLE = N / T.
    arrival_times must be increasing timestamps (floats or datetimes converted to seconds).
    """
    t = np.asarray(arrival_times)
    if t.size < 2:
        return 0.0
    T = float(t.max() - t.min()) or 1.0
    lam = (t.size - 1) / T
    return float(lam)


def fit_claim_distribution(amounts: np.ndarray, dist: str = "exponential") -> dict:
    """Fit claim size distribution params by MLE.
    Supports 'exponential', 'pareto', 'lognormal'. Returns dict with params and loglik.
    """
    x = np.asarray(amounts, dtype=float)
    if dist == "exponential":
        # Exponential scale = 1/lambda
        loc, scale = 0.0, float(np.mean(x) if x.size else 1.0)
        ll = np.sum(stats.expon.logpdf(x, loc=0, scale=scale))
        return {"dist": "exponential", "scale": scale, "loglik": float(ll)}
    elif dist == "pareto":
        # Using scipy's Pareto with shape b and scale
        b, loc, scale = stats.pareto.fit(x, floc=0)
        ll = np.sum(stats.pareto.logpdf(x, b, loc=loc, scale=scale))
        return {"dist": "pareto", "b": float(b), "scale": float(scale), "loglik": float(ll)}
    elif dist == "lognormal":
        s, loc, scale = stats.lognorm.fit(x, floc=0)
        ll = np.sum(stats.lognorm.logpdf(x, s, loc=loc, scale=scale))
        return {"dist": "lognormal", "s": float(s), "scale": float(scale), "loglik": float(ll)}
    else:
        raise ValueError("unknown dist")
