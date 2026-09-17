"""monte-carlo：蒙特卡洛模拟与可视化。

模块划分（单向依赖）：

    paths ─┐
           ├─→ option ─→ convergence ─→ plotting
           ┘                              ↑
    cli 负责串联 ─────────────────────────┘

核心思想：**用随机模拟逼近解析解**，并用标准误量化"逼近得有多好"。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
