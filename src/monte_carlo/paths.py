"""几何布朗运动（GBM）路径模拟。

风险中性测度下：$dS_t = r S_t dt + \\sigma S_t dW_t$，解析解

    S_t = S_0 exp( (r - σ²/2) t + σ W_t )

本模块直接**精确模拟**该解析解（而非 Euler 离散化），因此没有离散化偏差。
"""

from __future__ import annotations

import numpy as np


def simulate_gbm(
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    steps: int = 252,
    n_paths: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """模拟 GBM 路径。

    返回形状 `(n_paths, steps + 1)` 的数组：第 0 列为初始价 $S_0$。

    参数
    ----
    s0      : 初始价格
    mu      : 年化漂移（风险中性定价时应取无风险利率 r）
    sigma   : 年化波动率
    T       : 期限（年）
    steps   : 每条的步数（252 ≈ 一年的交易日）
    n_paths : 路径条数
    seed    : 随机种子（固定以保证可复现）
    """
    if steps <= 0 or n_paths <= 0:
        raise ValueError("steps 与 n_paths 必须为正整数")
    if sigma < 0:
        raise ValueError("sigma 不能为负")

    rng = np.random.default_rng(seed)
    dt = T / steps

    # 对数收益增量：(mu - σ²/2) dt + σ √dt · Z
    z = rng.standard_normal((n_paths, steps))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

    log_paths = np.cumsum(increments, axis=1)
    zeros = np.zeros((n_paths, 1))
    return s0 * np.exp(np.concatenate([zeros, log_paths], axis=1))


def terminal_values(paths: np.ndarray) -> np.ndarray:
    """取每条路径的终值 $S_T$。"""
    return paths[:, -1]


def time_grid(T: float, steps: int) -> np.ndarray:
    """时间网格 0, dt, ..., T。"""
    return np.linspace(0.0, T, steps + 1)
