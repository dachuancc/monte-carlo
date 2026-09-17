"""收敛性研究：蒙特卡洛价格随样本数 $N$ 如何逼近解析解。

理论：独立样本下估计量的标准误 $\\mathrm{SE} \\propto 1/\\sqrt{N}$，
即要多降一个数量级的误差，样本量得翻 100 倍——这是蒙特卡洛的**根本代价**。
"""

from __future__ import annotations

from dataclasses import dataclass

from .option import bs_price, mc_european

DEFAULT_N_GRID = (100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000)


@dataclass
class ConvergencePoint:
    n: int
    price: float
    stderr: float


@dataclass
class ConvergenceResult:
    points: list[ConvergencePoint]
    bs_price: float

    @property
    def ns(self) -> list[int]:
        return [p.n for p in self.points]

    @property
    def prices(self) -> list[float]:
        return [p.price for p in self.points]

    @property
    def stderrs(self) -> list[float]:
        return [p.stderr for p in self.points]


def convergence_study(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    kind: str = "call",
    n_grid: tuple[int, ...] = DEFAULT_N_GRID,
    antithetic: bool = True,
    seed: int = 0,
) -> ConvergenceResult:
    """在给定样本量网格上重复定价，观察收敛过程。"""
    points: list[ConvergencePoint] = []
    for n in n_grid:
        result = mc_european(
            s0=s0, k=k, r=r, sigma=sigma, T=T, n=n, kind=kind,
            antithetic=antithetic, seed=seed,
        )
        points.append(ConvergencePoint(n=result.n, price=result.price, stderr=result.stderr))

    return ConvergenceResult(
        points=points,
        bs_price=bs_price(s0=s0, k=k, r=r, sigma=sigma, T=T, kind=kind),
    )
