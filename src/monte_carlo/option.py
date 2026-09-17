"""欧式期权：Black–Scholes 解析解 + 蒙特卡洛定价。

- 解析解只用 `math`，无 scipy 依赖（正态 CDF 用 `erf` 实现）。
- 蒙特卡洛在风险中性测度下贴现期望收益，并给出**标准误**。
- 支持**对偶变量法**（antithetic variates）做方差缩减。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# 正态分布
# --------------------------------------------------------------------------- #
def norm_cdf(x: float) -> float:
    """标准正态分布函数（用 erf 实现，避免引入 scipy）。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Black–Scholes 解析解
# --------------------------------------------------------------------------- #
def bs_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    kind: str = "call",
) -> float:
    """欧式期权 Black–Scholes 价格。"""
    if kind not in {"call", "put"}:
        raise ValueError("kind 必须是 'call' 或 'put'")
    if T <= 0:
        # 到期：内在价值
        return max(s0 - k, 0.0) if kind == "call" else max(k - s0, 0.0)

    d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if kind == "call":
        return s0 * norm_cdf(d1) - k * math.exp(-r * T) * norm_cdf(d2)
    return k * math.exp(-r * T) * norm_cdf(-d2) - s0 * norm_cdf(-d1)


# --------------------------------------------------------------------------- #
# 蒙特卡洛定价
# --------------------------------------------------------------------------- #
@dataclass
class MCResult:
    """蒙特卡洛定价结果。

    price    : 期权价格估计
    stderr   : 估计的标准误（price 的 1σ 不确定度）
    n        : 使用的**路径数**（对偶变量法下 = 2×i.i.d. 对数）
    samples  : 用于估计的贴现收益样本（i.i.d.）
    method   : "naive" 或 "antithetic"
    """

    price: float
    stderr: float
    n: int
    samples: np.ndarray
    method: str

    @property
    def ci95(self) -> tuple[float, float]:
        """95% 置信区间（正态近似）。"""
        half = 1.96 * self.stderr
        return (self.price - half, self.price + half)


def _payoff(st: np.ndarray, k: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(st - k, 0.0)
    return np.maximum(k - st, 0.0)


def mc_european(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    n: int = 100_000,
    kind: str = "call",
    antithetic: bool = True,
    seed: int | None = None,
) -> MCResult:
    """用蒙特卡洛给欧式期权定价（风险中性测度）。

    对偶变量法下，把每对 $(Z, -Z)$ 的收益**先平均**再统计——
    这样得到的样本是 i.i.d. 的，标准误才是正确的（见 docs/DECISIONS.md D3）。
    """
    if kind not in {"call", "put"}:
        raise ValueError("kind 必须是 'call' 或 'put'")
    if n <= 1:
        raise ValueError("n 必须大于 1")

    rng = np.random.default_rng(seed)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    if antithetic:
        m = n // 2
        z = rng.standard_normal(m)
        st_plus = s0 * np.exp(drift + vol * z)
        st_minus = s0 * np.exp(drift - vol * z)
        # 每对先平均 → m 个 i.i.d. 样本（路径数仍为 2m）
        pair_payoff = 0.5 * (_payoff(st_plus, k, kind) + _payoff(st_minus, k, kind))
        samples = discount * pair_payoff
        n_paths = 2 * m
        method = "antithetic"
    else:
        z = rng.standard_normal(n)
        st = s0 * np.exp(drift + vol * z)
        samples = discount * _payoff(st, k, kind)
        n_paths = n
        method = "naive"

    price = float(samples.mean())
    stderr = float(samples.std(ddof=1) / math.sqrt(len(samples))) if len(samples) > 1 else 0.0
    return MCResult(price=price, stderr=stderr, n=n_paths, samples=samples, method=method)
