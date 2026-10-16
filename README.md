# XTX / SANOS 波动率曲面

本项目用 SPXW 收盘报价复现 SANOS 的数据处理、联合 LP 校准、无套利约束和连续曲面，
并研究广义增量模型、核心区域优先校准、跨日 QP、DLV 和状态依赖转移。

从仓库根目录的 `notebooks/project_overview.ipynb` 开始阅读：它依次可视化标准化、
筛选后的报价、方差骨架、价格与 IV 拟合、概率密度、多日验证、翼部对照和延伸探索。
方法和研究结论见 [项目总结](docs/SUMMARY.md)。展示读取已有结果，不会自动重新校准。

## 结构

```text
data/outputs/                 本地结果、缓存、图表，不纳入 Git
docs/SUMMARY.md               方法、验证结果与拓展研究总结
scripts/                      校准、查询、画图和诊断入口
src/                          SANOS 模型、曲面与拓展模块
notebooks/project_overview.ipynb  分步骤可视化展示
tests/                        回归测试
native/                       C++ 实验
```

## 运行

从仓库根目录执行。默认行情为 `D:\Projects\data\options_data\spx_index_2025.parquet`，
利率目录为 `D:\Projects\data\yield_data`；基础入口支持 `--data-path`、`--yield-dir` 覆盖。
原始数据不分发；生成结果统一保存到 `data/outputs/`。

```powershell
python -m pip install -r requirements.txt
python scripts/run_xtx_baseline.py --trade-date 2025-06-02 --eta 0.25
python scripts/query_xtx_surface.py --model-dir data/outputs/baseline_paper_gap001/2025-06-02 --dte 7 14 --k 0.95 1 1.03
python scripts/plot_xtx_calibration_quotes.py
python -m pytest -q
```

数据和展示结果需在本地提供；新克隆不会自动获得被忽略的 `data/outputs/`。
基础入口只运行基础校准，图表只显示参与校准的报价。
数据准备由 `src/xtx_data.py` 实现，公共结果加载与诊断核心定义位于 `src/xtx_inputs.py`。
正式校准不读取或执行 notebook。

| 用途 | 代码 |
| --- | --- |
| LP、定价核和凸序 | `src/sanos_core.py` |
| 数据准备与公共输入 | `src/xtx_data.py`、`src/xtx_inputs.py` |
| 连续曲面和审计 | `src/sanos_surface.py` |
| 原生 HiGHS 和热更新 | `src/sanos_incremental.py` |
| 跨日 QP | `src/sanos_daily.py` |
| 广义增量 | `src/sanos_generalized.py`、`src/sanos_generalized_grid.py` |
| 状态依赖转移 | `src/sanos_martingale_transport.py` |

`scripts/run_*.py` 是运行入口，`diagnose_*.py` 是诊断，`benchmark_*.py` 是性能实验。
无需依次执行全部脚本。

## 数据和输出

行情为 OptionMetrics IvyDB 的 SPXW 收盘期权数据，包含到期日、执行价、类型、
买卖价、成交量等字段。利率来自 DGS1MO、DGS3MO、DGS6MO、DGS1、DGS2 CSV。
原始数据不随仓库分发，也不需要复制到仓库内；`--data-path` 和 `--yield-dir` 可指定路径。
收盘日成交量只能用于收盘研究，不能回填到当天更早的盘中决策。

| 输出目录 | 内容 |
| --- | --- |
| `data/outputs/baseline_paper_gap001/<日期>/` | 基础校准默认结果 |
| `data/outputs/daily_validation/` | 9天验证及跨日实验，展示 notebook 的主要输入 |
| `data/outputs/calibration_quotes/` | 只显示拟合报价的交互 HTML |
| `data/outputs/wing_turns/` | 翼部形状扫描 |
| `data/outputs/wing_boundary/` | 固定报价的左尾状态对照 |
| `data/outputs/generalized_grid/`、`data/outputs/generalized_grid_core_first/` | 广义模型原目标及核心优先运行结果 |
| `data/outputs/research_exploration/` | 拟合失败归因、核心优先、QP 等探索对照 |

每个基础校准日期输出 `quotes.csv`（报价、拟合及误差）、`model.npz`（模型）、
`summary.json`（配置与审计）、`per_expiry.csv`、凸序／日历检查及覆盖率图。
`data/outputs/` 不纳入 Git；新克隆需生成或自行提供合法结果文件。

## 多日复现和研究入口

```powershell
python scripts/run_xtx_baseline.py --trade-date 2025-01-02 2025-01-03 2025-01-06 --eta 0.25 --output-dir data/outputs/formal_baseline
python scripts/run_xtx_baseline.py --trade-date 2025-04-01 2025-04-02 2025-04-03 --eta 0.1 --output-dir data/outputs/formal_baseline
python scripts/run_xtx_baseline.py --trade-date 2025-06-02 2025-06-03 2025-06-04 --eta 0.25 --output-dir data/outputs/formal_baseline
```

基础校准使用 SciPy 的 HiGHS 接口，跨日更新使用 highspy；两者均已包含在 `requirements.txt`，统一安装即可。

| 用途 | 入口 |
| --- | --- |
| 连续交易日验证 | `scripts/run_xtx_daily_validation.py` |
| 跨日缩减状态 QP | `scripts/run_xtx_daily_qp.py` |
| 广义增量 | `scripts/run_xtx_generalized_grid.py` |
| 状态转移 | `scripts/run_xtx_martingale_transport.py`、`scripts/run_xtx_transport_chain.py` |
| DLV | `scripts/run_xtx_dlv.py`、`scripts/run_xtx_dlv_calibration.py` |
| 翼部检查 | `scripts/diagnose_xtx_wing_turns.py`、`scripts/diagnose_xtx_wing_boundary.py` |
| 市场报价日历诊断 | `scripts/generate_xtx_market_calendar_diagnostics.py` |
| 联合 LP eta 敏感性 | `scripts/generate_xtx_multi_day_eta_regime_study.py` |
| 广义模型精确参考验证 | `scripts/validate_xtx_generalized.py` |
| 广义模型核心优先对照 | `scripts/explore_xtx_core_priority.py` |
| 广义拟合失败归因 | `scripts/diagnose_xtx_fit_failure.py` |
| 逐报价保护 QP | `scripts/explore_xtx_fit_preserving_qp.py`，由研究脚本调用 |

研究程序的参数可通过 `--help` 查看。展示 notebook 默认读取已有9天实验的6月2日结果；
若只有新校准结果，将首个代码单元的 `MODEL_DIR` 改为对应模型目录。
`scripts/build_project_overview.py` 可重新生成 notebook 源文件，但会清空其原有输出。

```powershell
python scripts/generate_xtx_market_calendar_diagnostics.py --date 2025-06-02
python scripts/generate_xtx_multi_day_eta_regime_study.py --dates 2025-06-02 --etas 0.1 0.25
python scripts/validate_xtx_generalized.py
```

对应输出为 `data/outputs/market_calendar/<日期>/`、`data/outputs/eta_sensitivity/`、
`data/outputs/generalized_accuracy/`。这些入口共用 `src/xtx_research.py`，不执行 notebook。

广义模型默认使用原论文目标；可显式启用核心优先，两阶段求解仍使用 LP。
以下命令分别运行一个日期、三天目标对照和失败归因。运行探索不更改基础 LP 默认设置。

```powershell
python scripts/run_xtx_generalized_grid.py --dates 2025-06-03 --grids 16385 --fit-priority core_first
python scripts/explore_xtx_core_priority.py
python scripts/diagnose_xtx_fit_failure.py
```

核心优先单次运行保存每期限 CSV、模型 NPZ 和 `summary.json`；目标对照保存
`data/outputs/research_exploration/core_priority/results.csv`、`results.json` 及各日期明细；
失败归因保存到 `data/outputs/research_exploration/failure_attribution/`。
`scripts/explore_xtx_*.py` 是延伸实验；其样本结果与限制统一记录在项目总结中。
