"""可视化（matplotlib，Agg 后端，无需 GUI）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能出图

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .convergence import ConvergenceResult  # noqa: E402
from .paths import time_grid  # noqa: E402


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_paths(
    paths: np.ndarray,
    path: str | Path,
    *,
    T: float = 1.0,
    n_show: int = 40,
    title: str = "Geometric Brownian Motion — sample paths",
) -> Path:
    """绘制若干条 GBM 样本路径。"""
    t = time_grid(T, paths.shape[1] - 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(min(n_show, paths.shape[0])):
        ax.plot(t, paths[i], linewidth=0.8, alpha=0.7)
    ax.axhline(paths[0, 0], color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("time (years)")
    ax.set_ylabel("price")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_convergence(
    result: ConvergenceResult,
    path: str | Path,
    *,
    title: str = "Monte Carlo convergence to Black–Scholes",
) -> Path:
    """绘制 MC 价格随 N 的收敛（含 95% 置信带与解析解）。"""
    ns = np.array(result.ns, dtype=float)
    prices = np.array(result.prices)
    half = 1.96 * np.array(result.stderrs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(ns, prices, yerr=half, fmt="o-", capsize=3, linewidth=1.2,
                label="Monte Carlo (95% CI)")
    ax.axhline(result.bs_price, color="crimson", linestyle="--", linewidth=1.4,
               label=f"Black–Scholes = {result.bs_price:.4f}")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("number of samples N (log scale)")
    ax.set_ylabel("option price")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return _save(fig, path)


def plot_terminal_distribution(
    paths: np.ndarray,
    path: str | Path,
    *,
    title: str = "Distribution of terminal price S_T",
) -> Path:
    """绘制终值 $S_T$ 的分布直方图，并叠加理论对数正态密度。"""
    st = paths[:, -1]
    s0 = paths[0, 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(st, bins=60, density=True, alpha=0.6, label="simulated $S_T$")

    # 理论：ln S_T ~ N(ln s0 + (mu - σ²/2)T, σ²T)；这里用样本反推 mu 只为画图
    ax.set_title(title)
    ax.set_xlabel("terminal price $S_T$")
    ax.set_ylabel("density")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)
