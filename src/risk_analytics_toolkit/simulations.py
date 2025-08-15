from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import optimize

from .utils import rng


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate geometric Brownian motion paths.
    S_{t+dt} = S_t * exp((mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z)
    Returns array shape (n_paths, steps+1).
    """
    g = rng(seed)
    paths = np.zeros((n_paths, steps + 1), dtype=float)
    paths[:, 0] = S0
    shock = g.normal(size=(n_paths, steps))
    inc = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * shock
    paths[:, 1:] = S0 * np.exp(np.cumsum(inc, axis=1))
    return paths


def simulate_cramer_lundberg(
    U0: float,
    premium_rate: float,
    claim_lambda: float,
    claim_dist: Callable[[np.random.Generator, int], np.ndarray],
    T: float,
    n_paths: int,
    seed: Optional[int] = None,
    dt: float = 1 / 252,
) -> np.ndarray:
    """Simulate reserve process U(t) = U0 + c t - sum_{i=1}^{N(t)} X_i.
    Poisson arrivals N(t) with rate lambda; claim sizes X_i ~ claim_dist.
    Return array (n_paths, steps+1).
    """
    g = rng(seed)
    steps = int(np.ceil(T / dt))
    U = np.zeros((n_paths, steps + 1), dtype=float)
    U[:, 0] = U0
    for t in range(1, steps + 1):
        # Poisson arrivals in dt
        N = g.poisson(claim_lambda * dt, size=n_paths)
        claims = np.zeros(n_paths)
        mask = N > 0
        if np.any(mask):
            # sum of N iid severities
            claims[mask] = np.array(
                [claim_dist(g, int(k)).sum() if k > 0 else 0.0 for k in N[mask]]
            )
        U[:, t] = U[:, t - 1] + premium_rate * dt - claims
    return U


def estimate_ruin_probability(simulated_paths: np.ndarray) -> float:
    """Estimate P(ruin) = fraction of paths that ever go <= 0."""
    U = np.asarray(simulated_paths)
    ruined = (U <= 0).any(axis=1)
    return float(np.mean(ruined))


def lundberg_upper_bound(
    premium_rate: float, claim_lambda: float, claim_mgf: Callable[[float], float]
) -> float:
    """Compute Lundberg bound exp(-theta * u) constant (returns theta*c/lambda approx).
    We solve E[e^{theta X}] = 1 + (theta c)/lambda for theta>0 via root-finding.
    This is a simplified form; real-world models differ. Use small fallback if fails.
    """

    def f(theta: float) -> float:
        return claim_mgf(theta) - (1.0 + theta * premium_rate / claim_lambda)

    try:
        theta = optimize.brentq(f, 1e-6, 10.0, maxiter=100)
        return float(theta)
    except Exception:
        return 0.0
