import numpy as np

from risk_analytics_toolkit.estimation import (
    fit_claim_distribution,
    fit_gbm_params,
    fit_poisson_intensity,
)


def test_fit_gbm_basic():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=1000)
    mu, sigma = fit_gbm_params(r)
    assert np.isfinite(mu) and np.isfinite(sigma)


def test_poisson_intensity():
    t = np.array([0, 1, 2, 3, 4])
    lam = fit_poisson_intensity(t)
    assert abs(lam - 1.0) < 1e-8


def test_claim_dist_exponential():
    rng = np.random.default_rng(0)
    x = rng.exponential(10.0, size=1000)
    res = fit_claim_distribution(x, dist="exponential")
    assert res["dist"] == "exponential"
    assert "scale" in res and res["scale"] > 0
