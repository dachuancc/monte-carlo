"""GBM 路径模拟测试。"""

from __future__ import annotations

import numpy as np

from monte_carlo.paths import simulate_gbm, terminal_values, time_grid


def test_shape():
    paths = simulate_gbm(n_paths=50, steps=100, seed=1)
    assert paths.shape == (50, 101)


def test_all_paths_start_at_s0():
    paths = simulate_gbm(s0=123.0, n_paths=20, steps=10, seed=1)
    assert np.allclose(paths[:, 0], 123.0)


def test_reproducible_with_seed():
    a = simulate_gbm(n_paths=10, steps=50, seed=99)
    b = simulate_gbm(n_paths=10, steps=50, seed=99)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    a = simulate_gbm(n_paths=10, steps=50, seed=1)
    b = simulate_gbm(n_paths=10, steps=50, seed=2)
    assert not np.array_equal(a, b)


def test_zero_volatility_is_deterministic_drift():
    """σ=0 时每条路径都应等于 S0·e^{μt}。"""
    paths = simulate_gbm(s0=100.0, mu=0.05, sigma=0.0, T=1.0, steps=4, n_paths=3, seed=1)
    t = time_grid(1.0, 4)
    expected = 100.0 * np.exp(0.05 * t)
    for row in paths:
        assert np.allclose(row, expected)


def test_terminal_lognormal_mean():
    """风险中性下 E[S_T] = S0·e^{μT}，用大样本近似验证。"""
    s0, mu, sigma, T = 100.0, 0.05, 0.2, 1.0
    paths = simulate_gbm(s0=s0, mu=mu, sigma=sigma, T=T, steps=252, n_paths=200_000, seed=11)
    st = terminal_values(paths)
    expected = s0 * np.exp(mu * T)
    # 相对误差在 1% 以内
    assert abs(st.mean() / expected - 1.0) < 0.01


def test_prices_positive():
    paths = simulate_gbm(n_paths=100, steps=50, seed=2)
    assert (paths > 0).all()
