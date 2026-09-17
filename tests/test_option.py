"""欧式期权定价测试：Black–Scholes 数值 + 蒙特卡洛收敛 + 方差缩减。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from monte_carlo.option import bs_price, mc_european, norm_cdf

# 教科书标准参数
S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
BS_CALL = 10.4506  # 已知解析值
BS_PUT = 5.5735


def test_norm_cdf_known_values():
    assert abs(norm_cdf(0.0) - 0.5) < 1e-12
    assert abs(norm_cdf(1.96) - 0.975) < 1e-3
    assert abs(norm_cdf(-1.96) - 0.025) < 1e-3


def test_bs_call_known_value():
    assert abs(bs_price(S0, K, R, SIGMA, T, "call") - BS_CALL) < 5e-3


def test_bs_put_known_value():
    assert abs(bs_price(S0, K, R, SIGMA, T, "put") - BS_PUT) < 5e-3


def test_put_call_parity():
    """C - P = S0 - K·e^{-rT}（精确关系，不是近似）。"""
    call = bs_price(S0, K, R, SIGMA, T, "call")
    put = bs_price(S0, K, R, SIGMA, T, "put")
    assert abs((call - put) - (S0 - K * math.exp(-R * T))) < 1e-9


def test_bs_deep_itm_call_approaches_intrinsic():
    """深度实值看涨 ≈ S0 - K·e^{-rT}。"""
    price = bs_price(200.0, 50.0, R, SIGMA, T, "call")
    assert price > 150.0


def test_bs_rejects_bad_kind():
    with pytest.raises(ValueError):
        bs_price(S0, K, R, SIGMA, T, "straddle")


def test_mc_converges_to_black_scholes():
    result = mc_european(S0, K, R, SIGMA, T, n=200_000, seed=42)
    # 误差应在 5 个标准误之内
    assert abs(result.price - BS_CALL) < 5 * result.stderr


def test_mc_reproducible_with_seed():
    a = mc_european(S0, K, R, SIGMA, T, n=10_000, seed=7)
    b = mc_european(S0, K, R, SIGMA, T, n=10_000, seed=7)
    assert a.price == b.price
    assert a.stderr == b.stderr


def test_antithetic_reduces_variance():
    """对偶变量法应降低看涨期权的估计方差。"""
    naive = mc_european(S0, K, R, SIGMA, T, n=100_000, antithetic=False, seed=1)
    anti = mc_european(S0, K, R, SIGMA, T, n=100_000, antithetic=True, seed=1)
    assert anti.stderr < naive.stderr


def test_standard_error_scales_like_inverse_sqrt_n():
    """样本量翻 4 倍，标准误应约减半（1/√N）。"""
    small = mc_european(S0, K, R, SIGMA, T, n=25_000, seed=3)
    large = mc_european(S0, K, R, SIGMA, T, n=100_000, seed=3)
    assert large.stderr < small.stderr * 0.75


def test_ci95_brackets_price():
    result = mc_european(S0, K, R, SIGMA, T, n=50_000, seed=5)
    lo, hi = result.ci95
    assert lo < result.price < hi


def test_mc_rejects_tiny_n():
    with pytest.raises(ValueError):
        mc_european(S0, K, R, SIGMA, T, n=1)
