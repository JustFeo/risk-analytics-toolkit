import numpy as np

from risk_analytics_toolkit.metrics import (
    cvar_historical,
    var_historical,
    var_parametric_normal,
)


def test_normal_parametric_var_matches_scipy():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 1.0, size=10000)
    v = var_parametric_normal(r, alpha=0.05)
    # quantile of loss at alpha ~ -norm.ppf(alpha)
    from scipy.stats import norm

    assert abs(v - (-norm.ppf(0.05))) < 1e-2


def test_hist_metrics_return_floats():
    r = np.array([0.0, 0.01, -0.02, 0.03, -0.01])
    assert isinstance(var_historical(r, 0.1), float)
    assert isinstance(cvar_historical(r, 0.1), float)
