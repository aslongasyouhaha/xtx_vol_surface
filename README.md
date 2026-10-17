# XTX / SANOS 波动率曲面

本项目用 SPXW 收盘报价复现 SANOS 的数据处理、联合 LP 校准、无套利约束和连续曲面，
并研究广义增量模型、核心区域优先校准、跨日 QP、DLV 和状态依赖转移。

从仓库根目录的 `notebooks/project_overview.ipynb` 开始阅读：它依次可视化标准化、
筛选后的报价、方差骨架、价格与 IV 拟合、概率密度、多日验证、翼部对照和延伸探索。
方法和研究结论见 [项目总结](docs/SUMMARY.md)。展示读取已有结果，不会自动重新校准。
多期限 IV 叠加图和三维曲面可在 VS Code / Jupyter 中交互查看；GitHub 预览不执行交互脚本。

参考论文：[SANOS（PDF）](docs/SANOS.pdf) · [arXiv:2601.11209v2](https://arxiv.org/abs/2601.11209v2)。

## 结构

```text
data/testcases/               可公开运行的合成报价、参数与预期结果
data/outputs/                 本地结果、缓存、图表，不纳入 Git
docs/SUMMARY.md               方法、验证结果与拓展研究总结
docs/SANOS.pdf                SANOS 参考论文（arXiv v2）
scripts/                      5个文件：基础校准、查询、研究、notebook 构建、启动辅助
src/                          16个模块：基础流程、延伸模型及研究实现
notebooks/project_overview.ipynb  分步骤可视化展示
tests/                        回归测试
native/                       C++ 实验
```

## 运行

从仓库根目录执行。先安装依赖，再运行仓库自带的三个合成案例，无需订阅行情：

```powershell
python -m pip install -r requirements.txt
python scripts/research_xtx.py testcases
python scripts/query_xtx_surface.py --model-dir data/outputs/testcases/clean --dte 30 45 60 --k 0.95 1 1.05
python scripts/research_xtx.py plot --model-dir data/outputs/testcases/clean --output data/outputs/testcases/clean/iv.html
python -m pytest -q
```

真实行情校准需要本地数据。默认行情为 `D:\Projects\data\options_data\spx_index_2025.parquet`，
利率目录为 `D:\Projects\data\yield_data`；基础入口支持 `--data-path`、`--yield-dir` 覆盖。

```powershell
python scripts/run_xtx_baseline.py --trade-date 2025-06-02 --eta 0.25
python scripts/query_xtx_surface.py --model-dir data/outputs/baseline_paper_gap001/2025-06-02 --dte 7 14 --k 0.95 1 1.03
python scripts/research_xtx.py plot --model-dir data/outputs/baseline_paper_gap001/2025-06-02
```

生成结果统一保存到 `data/outputs/`；新克隆不会自动获得这个被忽略的目录。
主展示 notebook 读取真实行情的已有结果，仍需自行生成或提供相应本地文件。
基础入口只运行基础校准，图表只显示参与校准的报价。
数据准备由 `src/xtx_data.py` 实现，公共输入输出与诊断核心定义位于 `src/xtx_io.py`。
正式校准不读取或执行 notebook。

| 用途 | 代码 |
| --- | --- |
| LP、定价核和凸序 | `src/sanos_core.py` |
| 数据准备与输入输出 | `src/vol_surface_utils.py`、`src/xtx_data.py`、`src/xtx_io.py` |
| 连续曲面和审计 | `src/sanos_surface.py` |
| 原生 HiGHS 和热更新 | `src/sanos_incremental.py` |
| 求解器对照与性能基准 | `src/sanos_solver_research.py`，工作集、活跃列和约束生成集中在此 |
| 跨日 QP | `src/sanos_daily.py` |
| 广义增量 | `src/sanos_generalized.py`，含精确路径参考和保守网格实现 |
| DLV 生成、恢复与校准 | `src/sanos_dlv.py` |
| 状态依赖转移 | `src/sanos_martingale_transport.py` |
| 研究实验 | `src/xtx_research.py`、`src/xtx_generalized_research.py`、`src/xtx_daily_research.py`、`src/xtx_transport_research.py` |
| 共用展示 | `src/xtx_plotting.py` |

`scripts/` 负责命令参数与运行，公共计算逻辑位于 `src/`，测试直接导入公共模块。
`scripts/` 只保留以下5个文件，研究功能通过子命令选择：

| 文件 | 用途 |
| --- | --- |
| `run_xtx_baseline.py` | 真实行情的基础联合 LP 校准 |
| `query_xtx_surface.py` | 查询保存曲面的价格 |
| `research_xtx.py` | 可视化、验证、延伸模型及求解器对照的统一入口 |
| `build_project_overview.py` | 构建展示 notebook 源文件 |
| `_bootstrap.py` | 为直接执行脚本配置项目路径与本地依赖 |

## 数据和输出

真实研究行情来自 **OptionMetrics IvyDB US，经 WRDS 平台获取**，使用其中的 SPXW 收盘期权报价。
包含到期日、执行价、类型、买卖价、成交量等字段；数据库和平台说明见
[WRDS 的 OptionMetrics 介绍](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/)。
利率来自 DGS1MO、DGS3MO、DGS6MO、DGS1、DGS2 CSV。

真实报价的公开再分发取决于订阅协议授权，注明来源本身不构成再分发许可。
本仓库不随附真实报价或其抽样；有数据访问权限的使用者可通过 `--data-path` 和 `--yield-dir`
指定本地路径。下面的公开 CSV 全部为合成数据。
收盘日成交量只能用于收盘研究，不能回填到当天更早的盘中决策。

### 公开 testcase

`data/testcases/` 包含三个小型 CSV；每个文件和输出报告都标记 `data_kind=synthetic`。
输入已经是标准化的 call-equivalent bid、ask、mid，覆盖30、60、90天三个期限。
这些案例用于验证定价核、联合 LP、凸序约束、连续期限查询和数值审计，
不覆盖原始 OptionMetrics 文件导入、远期估计或成交量筛选。

| 案例 | 数据 | 预期行为 |
| --- | --- | --- |
| `clean` | 63条相容报价，由已知混合分布生成 | 全部价格落入 bid–ask，曲面审计通过 |
| `wide_wings` | 相同中间价，扩大翼部价差 | 全部价格落入 bid–ask，曲面审计通过 |
| `conflicting_quotes` | 64条报价，同一个30天 ATM 合约有两个不相交报价区间 | 求解及曲面审计通过，但无法全部拟合；区间外损失至少36.5个价差单位 |

[cases.json](data/testcases/cases.json) 记录生成参数、状态、概率权重、核方差和预期结果。
报价由这些参数解析计算，未复制 WRDS、OptionMetrics 或 CRSP 的数据记录。
`2000-01-03` 是合成案例的日期标签，不代表当日真实市场。
可用 `python scripts/research_xtx.py testcases --cases clean` 只运行一个案例。

第三个案例有意保留报价冲突，用于检查程序能区分“求解完成”和“所有报价拟合成功”。
每个案例输出 `quotes.csv`、`model.npz`、`summary.json`，总览在
`data/outputs/testcases/results.json`；`testcase_passed` 表示结果符合该案例预期，
`fit_target_met` 表示所有报价落在 bid–ask 内。

| 输出目录 | 内容 |
| --- | --- |
| `data/outputs/testcases/<案例>/` | 合成案例的模型、重定价与数值审计 |
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

真实行情可先选三个代表日期复现；这些参数和已有研究结果不依赖公开合成案例：

| 日期 | eta | 期限数／报价数 | 已有全报价覆盖率 |
| --- | --- | --- | --- |
| 2025-01-02 | 0.25 | 12／802 | 99.127% |
| 2025-04-01 | 0.1 | 13／869 | 100.000% |
| 2025-06-02 | 0.25 | 9／660 | 100.000% |

三天的诊断核心覆盖率均为100%；定义和完整9天结果见 [项目总结](docs/SUMMARY.md)。
运行下面命令可复现全部9天；将 `--trade-date` 保留为单个日期即可单独运行：

```powershell
python scripts/run_xtx_baseline.py --trade-date 2025-01-02 2025-01-03 2025-01-06 --eta 0.25 --output-dir data/outputs/formal_baseline
python scripts/run_xtx_baseline.py --trade-date 2025-04-01 2025-04-02 2025-04-03 --eta 0.1 --output-dir data/outputs/formal_baseline
python scripts/run_xtx_baseline.py --trade-date 2025-06-02 2025-06-03 2025-06-04 --eta 0.25 --output-dir data/outputs/formal_baseline
```

基础校准使用 SciPy 的 HiGHS 接口，跨日更新使用 highspy；两者均已包含在 `requirements.txt`，统一安装即可。

| 用途 | 入口 |
| --- | --- |
| 连续交易日验证 | `research_xtx.py daily-validation` |
| 跨日缩减状态 QP | `research_xtx.py daily-update` |
| 广义增量 | `research_xtx.py generalized` |
| 状态转移 | `research_xtx.py transport joint\|chain\|prefix` |
| DLV | `research_xtx.py dlv-recovery`、`dlv-calibrate` |
| 同一论文 LP 的固定几何更新基准 | `research_xtx.py benchmark --mode paper` |
| Python／C++ 冷启动与热更新基准 | `research_xtx.py benchmark --mode language` |

研究程序的参数可通过 `--help` 查看。展示 notebook 默认读取已有9天实验的6月2日结果；
若只有新校准结果，将首个代码单元的 `MODEL_DIR` 改为对应模型目录。
`scripts/build_project_overview.py` 可重新生成 notebook 源文件，但会清空其原有输出。

```powershell
python scripts/research_xtx.py --help
python scripts/research_xtx.py calendar --date 2025-06-02
python scripts/research_xtx.py eta --dates 2025-06-02 --etas 0.1 0.25
python scripts/research_xtx.py accuracy
```

对应输出为 `data/outputs/market_calendar/<日期>/`、`data/outputs/eta_sensitivity/`、
`data/outputs/generalized_accuracy/`。所有实验都使用公共输入与模型，不执行 notebook。

| `research_xtx.py` 子命令 | 内容 |
| --- | --- |
| `plot` | 参与校准的报价与拟合 IV 交互图，`--output` 指定 HTML 文件 |
| `generalized`、`transport` | 广义独立增量、三种状态转移对照 |
| `dlv-recovery`、`dlv-calibrate` | DLV 生成恢复检查、非线性校准 |
| `daily-validation`、`daily-update` | 连续交易日验证、以前一接受结果为参考的 QP 更新 |
| `benchmark` | 固定几何上的更新基准，选择 `--mode paper` 或 `language` |
| `testcases` | 无需真实行情的三个公开合成案例 |
| `calendar`、`eta` | 市场报价日历诊断、固定网格的 eta 敏感性 |
| `wings`、`wing-boundary` | 保存曲面的翼部扫描、共同左尾状态对照 |
| `surface` | 保存模型的连续插值、重定价及无套利审计 |
| `accuracy`、`refine-grid` | 精确路径参考、固定参数的细网格重定价 |
| `core-priority`、`fit-failure` | 核心优先目标对照、广义拟合失败归因 |
| `increment-flexibility`、`prefix-lookback` | 独立增量字典、相邻期限局部回调 |
| `daily-qp` | 逐报价保护 QP，可选择变量坐标 |

每个子命令支持 `--help` 和 `--output`，例如 `python scripts/research_xtx.py daily-qp --help`。
`daily-qp` 使用前日 LP 作隔离对照；`daily-update` 使用先前接受的结果作连续更新。
`daily-update --baseline-dir` 指定已有的连续日期 LP 结果根目录，`--output` 可另存更新结果。
`dlv-recovery --synthetic-only` 可在没有真实行情的情况下检查生成与恢复。

求解器对照代码集中维护，基础 LP 仍使用已有默认求解流程。`benchmark --mode paper`
使用当前论文目标；`--mode language` 保留仅区间外损失的受控 Python／C++ 对照，
需要先通过 `native/build.ps1` 构建本地程序。两种基准的目标不同，速度结果分别解释。

广义模型默认使用原论文目标；可显式启用核心优先，两阶段求解仍使用 LP。
以下命令分别运行一个日期、三天目标对照和失败归因。运行探索不更改基础 LP 默认设置。

```powershell
python scripts/research_xtx.py generalized --dates 2025-06-03 --grids 16385 --fit-priority core_first
python scripts/research_xtx.py core-priority
python scripts/research_xtx.py fit-failure
python scripts/research_xtx.py daily-qp --blocks june --coordinates absolute
python scripts/research_xtx.py transport chain --dates 2025-06-02
python scripts/research_xtx.py dlv-recovery --synthetic-only
```

核心优先单次运行保存每期限 CSV、模型 NPZ 和 `summary.json`；目标对照保存
`data/outputs/research_exploration/core_priority/results.csv`、`results.json` 及各日期明细；
失败归因保存到 `data/outputs/research_exploration/failure_attribution/`。
逐报价保护 QP 默认保存到 `data/outputs/research_exploration/fit_preserving_qp/`。
转移实验的 `joint`、`chain`、`prefix` 分别表示联合边际后耦合、逐期限递推、固定广义前缀对照；
默认输出在 `data/outputs/generalized_grid/` 下对应的 `joint_transport/`、`transport_chain/`、
`state_dependent_transport/<日期>/`。样本结果与限制统一记录在项目总结中。
