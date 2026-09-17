"""命令行入口。

    monte-carlo paths --n-paths 40 --plot docs/paths.png
    monte-carlo price --kind call --n 200000
    monte-carlo convergence --n-max 300000 --plot docs/convergence.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .convergence import DEFAULT_N_GRID, convergence_study
from .option import bs_price, mc_european
from .paths import simulate_gbm


def _run_paths(args: argparse.Namespace) -> int:
    paths = simulate_gbm(
        s0=args.s0, mu=args.mu, sigma=args.sigma, T=args.T,
        steps=args.steps, n_paths=args.n_paths, seed=args.seed,
    )
    st = paths[:, -1]
    print(f"路径数: {paths.shape[0]}   步数: {paths.shape[1] - 1}   初始价: {args.s0}")
    print(f"S_T 均值: {st.mean():.4f}   S_T 标准差: {st.std(ddof=1):.4f}")
    print(f"S_T 分位: 5%={np.percentile(st, 5):.2f}  50%={np.percentile(st, 50):.2f}  95%={np.percentile(st, 95):.2f}")
    if args.plot:
        from .plotting import plot_paths

        out = plot_paths(paths, args.plot, T=args.T, n_show=args.n_show)
        print(f"已保存: {out}")
    return 0


def _run_price(args: argparse.Namespace) -> int:
    result = mc_european(
        s0=args.s0, k=args.k, r=args.r, sigma=args.sigma, T=args.T,
        n=args.n, kind=args.kind, antithetic=not args.no_antithetic, seed=args.seed,
    )
    analytic = bs_price(args.s0, args.k, args.r, args.sigma, args.T, args.kind)
    lo, hi = result.ci95
    diff = result.price - analytic

    print(f"欧式期权: {args.kind}   S0={args.s0}  K={args.k}  r={args.r}  σ={args.sigma}  T={args.T}")
    print(f"方法: {result.method}   路径数: {result.n:,}")
    print("-" * 46)
    print(f"蒙特卡洛价格 : {result.price:.6f}   (±{result.stderr:.6f} 1σ)")
    print(f"95% 置信区间 : [{lo:.6f}, {hi:.6f}]")
    print(f"Black–Scholes: {analytic:.6f}")
    print(f"误差         : {diff:+.6f}   (|误差| / SE = {abs(diff) / result.stderr:.2f})")
    return 0


def _run_convergence(args: argparse.Namespace) -> int:
    grid = tuple(g for g in DEFAULT_N_GRID if g <= args.n_max)
    result = convergence_study(
        s0=args.s0, k=args.k, r=args.r, sigma=args.sigma, T=args.T,
        kind=args.kind, n_grid=grid, antithetic=not args.no_antithetic, seed=args.seed,
    )
    print(f"Black–Scholes 解析解: {result.bs_price:.6f}")
    print(f"{'N':>10} {'MC 价格':>12} {'标准误':>12} {'误差':>12} {'误差/SE':>10}")
    print("-" * 62)
    for p in result.points:
        err = p.price - result.bs_price
        ratio = abs(err) / p.stderr if p.stderr else 0.0
        print(f"{p.n:>10,} {p.price:>12.6f} {p.stderr:>12.6f} {err:>+12.6f} {ratio:>10.2f}")
    if args.plot:
        from .plotting import plot_convergence

        out = plot_convergence(result, args.plot)
        print(f"已保存: {out}")
    return 0


def _common_market(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--s0", type=float, default=100.0, help="初始价格")
    parser.add_argument("--k", type=float, default=100.0, help="行权价")
    parser.add_argument("--r", type=float, default=0.03, help="无风险利率")
    parser.add_argument("--sigma", type=float, default=0.2, help="年化波动率")
    parser.add_argument("--T", type=float, default=1.0, help="期限（年）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="monte-carlo", description="蒙特卡洛模拟与可视化")
    sub = parser.add_subparsers(dest="command", required=True)

    p_paths = sub.add_parser("paths", help="模拟并绘制 GBM 路径")
    _common_market(p_paths)
    p_paths.add_argument("--mu", type=float, default=0.05, help="年化漂移")
    p_paths.add_argument("--steps", type=int, default=252)
    p_paths.add_argument("--n-paths", type=int, default=100)
    p_paths.add_argument("--n-show", type=int, default=40, help="图上画多少条")
    p_paths.add_argument("--plot", metavar="PATH")
    p_paths.set_defaults(func=_run_paths)

    p_price = sub.add_parser("price", help="给欧式期权定价")
    _common_market(p_price)
    p_price.add_argument("--kind", choices=["call", "put"], default="call")
    p_price.add_argument("--n", type=int, default=200_000, help="样本数")
    p_price.add_argument("--no-antithetic", action="store_true", help="关闭对偶变量法")
    p_price.set_defaults(func=_run_price)

    p_conv = sub.add_parser("convergence", help="收敛性研究")
    _common_market(p_conv)
    p_conv.add_argument("--kind", choices=["call", "put"], default="call")
    p_conv.add_argument("--n-max", type=int, default=100_000)
    p_conv.add_argument("--no-antithetic", action="store_true")
    p_conv.add_argument("--plot", metavar="PATH")
    p_conv.set_defaults(func=_run_convergence)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
