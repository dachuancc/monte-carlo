"""收敛性研究的结构测试。"""

from __future__ import annotations

from monte_carlo.convergence import convergence_study


def _run():
    return convergence_study(
        s0=100.0, k=100.0, r=0.05, sigma=0.2, T=1.0,
        n_grid=(100, 1_000, 10_000), seed=0,
    )


def test_points_match_grid():
    result = _run()
    assert result.ns == [100, 1_000, 10_000]


def test_bs_price_is_reference():
    result = _run()
    assert abs(result.bs_price - 10.4506) < 5e-3


def test_standard_error_decreases_with_n():
    result = _run()
    stderrs = result.stderrs
    assert stderrs[0] > stderrs[1] > stderrs[2]


def test_prices_get_closer_overall():
    """大样本的误差应小于最小样本的误差（弱断言，避免随机性导致的偶发失败）。"""
    result = _run()
    err_small = abs(result.prices[0] - result.bs_price)
    err_large = abs(result.prices[-1] - result.bs_price)
    assert err_large < err_small
