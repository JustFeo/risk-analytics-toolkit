import numpy as np

from risk_analytics_toolkit.simulations import estimate_ruin_probability, simulate_gbm


def test_gbm_shape_and_seed():
    paths1 = simulate_gbm(100.0, 0.05, 0.2, dt=1 / 252, steps=10, n_paths=3, seed=42)
    paths2 = simulate_gbm(100.0, 0.05, 0.2, dt=1 / 252, steps=10, n_paths=3, seed=42)
    assert paths1.shape == (3, 11)
    assert np.allclose(paths1, paths2)


def test_ruin_probability_zero_if_all_positive():
    U = np.ones((5, 10))
    assert estimate_ruin_probability(U) == 0.0
