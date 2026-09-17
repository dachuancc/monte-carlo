# AGENTS.md — monte-carlo

给 AI 编码助手（pi 等）的项目上下文。**开始改动前，先读 `docs/ROADMAP.md`（进度与下一步）
和 `docs/DECISIONS.md`（已定的架构决策，勿轻易推翻）。**

## 这是什么

展示**蒙特卡洛方法**的小型工具库：GBM 路径模拟、欧式期权定价、收敛性研究、方差缩减。
重点不是"能算期权"，而是**把"估计得准不准"量化出来**（标准误 / 置信区间）。

## 技术栈

- Python 3.11+ · NumPy · matplotlib
- 依赖与虚拟环境用 `uv` 管理；测试用 pytest；CI 用 GitHub Actions
- **不依赖 scipy**（正态 CDF 用 `math.erf` 自实现）

## 目录结构

```
src/monte_carlo/
├── paths.py        # GBM 路径模拟（精确解，非 Euler 离散）
├── option.py       # Black–Scholes 解析解 + MC 定价 + 对偶变量法
├── convergence.py  # 收敛性研究（SE ∝ 1/√N）
├── plotting.py     # 路径图 / 收敛图 / 分布图（Agg 后端）
└── cli.py          # 命令行入口（paths / price / convergence）
tests/              # 单元测试
scripts/            # make_readme_assets.py
```

## 常用命令

```bash
uv sync
uv run pytest
uv run monte-carlo price --kind call --n 200000
uv run monte-carlo convergence --n-max 300000 --plot docs/convergence.png
uv run python scripts/make_readme_assets.py
```

## 代码约定

- **改动必须带测试**，且 `uv run pytest` 全绿才算完成。
- 任何定价函数**必须返回标准误**（`MCResult.stderr`），不准只返回一个点估计。
- 涉及随机数的地方**必须支持 `seed`**，保证可复现、测试稳定。
- **提交信息**：用中文，见下。

## 提交信息规范

| 类别 | 格式 | 例子 |
|---|---|---|
| 里程碑功能 | `M<编号>: 简述` | `M3: 新增亚式期权定价` |
| 其他 | `<类型>: 简述`（`文档`/`工具`/`配置`/`修复`/`测试`/`重构`） | `文档: 补充方差缩减对比` |

- 用中文，主题行简短，不以句号结尾；一次提交只做一件事。

## 当前状态

见 `docs/ROADMAP.md`「当前状态」——唯一状态真相。测试基线 **23 个用例**。

## 红线

- **风险中性漂移**：定价模拟的漂移必须是无风险利率 $r$，不是真实收益率 $\mu$。改错会得到错误的"价格"。
- **对偶变量法的标准误**：先把每对收益平均再统计；把相关样本当独立样本会**低估**标准误。
- **不虚构结果**：README / 文档里的数字必须由 `scripts/make_readme_assets.py` 生成，不得手改。
- **示例即示例**：所有结果都是教科书参数的合成模拟，不得包装成真实市场或投资结论。
