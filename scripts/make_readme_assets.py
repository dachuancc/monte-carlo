"""生成 README 用的素材：路径图 + 收敛图 + 结果表。

用法：uv run python scripts/make_readme_assets.py
产物：docs/sample_paths.png、docs/convergence.png，并打印指标表。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monte_carlo.convergence import convergence_study  # noqa: E402
from monte_carlo.option import mc_european  # noqa: E402
from monte_carlo.paths import simulate_gbm  # noqa: E402
from monte_carlo.plotting import plot_convergence, plot_paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

S0, K, R, SIGMA, T = 100.0, 100.0, 0.03, 0.2, 1.0


def main() -> None:
    # ---- 路径图 ----
    paths = simulate_gbm(s0=S0, mu=R, sigma=SIGMA, T=T, steps=252, n_paths=200, seed=42)
    out_paths = plot_paths(paths, DOCS / "sample_paths.png", T=T, n_show=40)
    print(f"[saved] {out_paths.relative_to(ROOT)}")

    # ---- 收敛图 ----
    result = convergence_study(
        s0=S0, k=K, r=R, sigma=SIGMA, T=T, seed=0,
        n_grid=(100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000),
    )
    out_conv = plot_convergence(result, DOCS / "convergence.png")
    print(f"[saved] {out_conv.relative_to(ROOT)}")

    # ---- 方差缩减对比 ----
    naive = mc_european(S0, K, R, SIGMA, T, n=200_000, antithetic=False, seed=1)
    anti = mc_european(S0, K, R, SIGMA, T, n=200_000, antithetic=True, seed=1)

    print()
    print(f"Black–Scholes 解析解: {result.bs_price:.6f}")
    print()
    print("| 方法 | 价格 | 标准误 | 方差缩减 |")
    print("|---|---|---|---|")
    print(f"| 朴素 MC | {naive.price:.6f} | {naive.stderr:.6f} | — |")
    print(f"| 对偶变量法 | {anti.price:.6f} | {anti.stderr:.6f} | {1 - anti.stderr / naive.stderr:.1%} |")
    print()
    print(f"标准误之比 (antithetic / naive): {anti.stderr / naive.stderr:.3f}")


if __name__ == "__main__":
    main()
