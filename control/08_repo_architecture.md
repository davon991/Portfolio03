# 08_repo_architecture.md

## 1. 文件使命

本文件冻结本项目的代码仓架构设计。
从本文件起，后续所有代码实现、数据处理、模型求解、实验运行、结果导出、论文图表生成和 GPT 二次分析，都必须遵守本文件规定的：

* 项目目录结构；
* 模块职责边界；
* 输入/输出契约；
* 命名规范；
* 配置文件规范；
* 结果目录组织方式；
* 与论文和总控文件的映射关系。

本文件的目标不是“搭一个能跑的杂乱代码仓”，而是搭一个**研究管线型仓库**：

> 总控文件 → 数据层 → 风险估计层 → 模型层 → 求解器层 → 回测层 → 导出层 → 论文资产层 → GPT 可读分析层

---

## 2. 架构总原则（冻结）

### 2.1 总控文件优先原则

代码仓不是自由生长的工程项目，必须服从以下总控文件：

* `00_project_charter.md`
* `01_notation_master.md`
* `02_definition_formula_ledger.md`
* `03_thesis_skeleton.md`
* `04_literature_map.md`
* `05_data_contract.md`
* `06_model_contract.md`
* `07_solver_contract.md`

所有代码模块不得自行改动：

* 记号；
* 定义；
* 变量名；
* 模型口径；
* 结果字段名；
* 参数角色。

### 2.2 单一研究管线原则

本仓库不允许“论文代码一套、分析代码一套、临时 notebook 一套、导图脚本一套”各自为政。
统一采用以下流水线：

1. 数据下载与清洗；
2. 风险估计；
3. 模型求解；
4. 回测；
5. 结果导出；
6. 论文图表 / 表格生成；
7. GPT 可读 JSON / CSV 分析资产生成。

### 2.3 一次 run，双重输出原则

每次正式实验运行只允许有一个 `run_id`，但必须同时输出两类结果：

#### A. 论文资产（paper assets）

* 可直接入论文的图表（PNG / PDF）
* 可直接入论文的表格（CSV / LaTeX）

#### B. GPT 可读资产（analysis assets）

* CSV / JSON / long-format panels
* solver diagnostics
* objective terms
* analysis pack

不允许出现“论文图来自 A 版本，GPT 分析来自 B 版本”的情况。

### 2.4 模块职责单一原则

每个模块只做一类工作，不允许一个脚本同时负责：

* 下载数据 + 跑回测 + 出图 + 写论文表；
* 或者在模型层中偷偷做数据清洗；
* 或者在图表层重新计算指标。

### 2.5 notebook 从属原则

Notebook 只能作为：

* 调试；
* 展示；
* 临时探索；

不得成为正式计算主线。
正式主线必须由脚本/模块驱动，并可命令行复现。

---

## 3. 仓库的功能分层

本仓库固定分成 8 层：

### 3.1 Control Layer（总控层）

保存所有冻结文档和契约。
角色：唯一真相源（single source of truth）。

### 3.2 Data Layer（数据层）

负责：

* ETF 原始价格下载（后续实现）；
* 对齐交易日历；
* 复权价格构造；
* 收益率生成；
* 数据审计与摘要输出。

### 3.3 Risk Layer（风险估计层）

负责：

* 滚动协方差矩阵估计；
* 相关矩阵估计；
* 波动率向量生成；
* 风险估计元数据输出。

### 3.4 Objective / Model Layer（目标与模型层）

负责：

* CtR 对象与 `D_R`；
* CtB 对象与 `D_B`；
* CtR-only 模型；
* CtB-only 模型；
* 主模型 objective closure。

### 3.5 Solver Layer（求解器层）

负责：

* 统一 solver family；
* projected gradient；
* active-set damped Newton；
* 投影与 KKT/诊断。

### 3.6 Backtest Layer（回测层）

负责：

* 月度调仓；
* 权重漂移；
* 组合收益累计；
* 换手计算；
* 各策略对照运行。

### 3.7 Export Layer（导出层）

负责：

* 结构化结果文件输出；
* 论文图表底层表导出；
* GPT 可读 analysis assets 导出。

### 3.8 Paper Asset Layer（论文资产层）

负责：

* 图表生成；
* LaTeX 表格生成；
* 论文图注/表注草稿生成（可选）。

---

## 4. 模块职责（冻结）

以下职责划分必须严格遵守。

### 4.1 `control/`

保存所有 markdown 契约文件。
职责：

* 不进行计算；
* 只定义规则；
* 是论文、代码、结果字段命名的唯一来源。

### 4.2 `src/data/`

职责：

* 读取原始价格；
* 复权与对齐；
* 构造日收益率；
* 构造月度调仓日历；
* 输出数据审计报告。

禁止：

* 估计协方差；
* 求解模型；
* 画图。

### 4.3 `src/risk/`

职责：

* 根据 `returns_daily.csv` 估计 `cov_t`、`corr_t`、`sigma_t`；
* 输出风险面板文件；
* 记录估计窗口与 shrinkage 参数。

禁止：

* 修改收益率口径；
* 直接调用求解器；
* 输出论文图。

### 4.4 `src/objectives/`

职责：

* 计算 `ctr`、`ctb`；
* 计算 `dr`、`db`；
* 构造 CtR-only / CtB-only / 主模型 objective；
* 提供梯度与 Hessian closure。

禁止：

* 处理数据缺失；
* 改变可行域定义；
* 负责 solver 迭代控制。

### 4.5 `src/solvers/`

职责：

* 实现 `solve_portfolio()` 顶层接口；
* 实现 projected gradient；
* 实现 active-set damped Newton；
* 实现 capped-simplex projection；
* 输出 solver diagnostics。

禁止：

* 重新定义 objective；
* 重新估计协方差；
* 在内部改变结果字段命名。

### 4.6 `src/backtest/`

职责：

* 按月度调仓循环所有策略；
* 生成目标权重；
* 计算权重漂移；
* 计算组合净值、换手、rolling 指标。

禁止：

* 修改单期求解器行为；
* 直接画论文图；
* 在回测脚本中临时改参数而不走配置文件。

### 4.7 `src/exports/`

职责：

* 将 run 结果导出成标准文件；
* 生成 `summary_metrics.csv`、`weights.csv`、`ctr_long.csv` 等；
* 生成 `diagnostics.json` 与 `analysis_pack.json`。

禁止：

* 重新计算主指标；
* 静默过滤异常日期；
* 改变字段名。

### 4.8 `src/figures/`

职责：

* 从标准结果文件生成论文图；
* 不直接调用求解器；
* 不直接读取原始行情。

禁止：

* 在画图时重新跑实验；
* 在画图时临时重算另一个口径的指标。

### 4.9 `src/tables/`

职责：

* 从标准结果文件生成论文表与 LaTeX 表；
* 保证表格可追溯到 `results/runs/<run_id>/` 下的 CSV/JSON。

### 4.10 `scripts/`

职责：

* 组织命令行入口；
* 串联数据、风险、模型、回测、导出、图表流程；
* 不承载核心业务逻辑。

---

## 5. I/O 契约（冻结）

### 5.1 数据层输入输出

#### 输入

* 原始价格文件或 API 下载结果（后续实现）
* `configs/universe.yaml`
* `05_data_contract.md`

#### 输出

* `data/processed/prices_daily_adj.parquet`
* `data/processed/returns_daily.csv`
* `data/processed/calendar_master.csv`
* `data/processed/universe_mask.csv`
* `data/processed/data_audit_report.json`
* `data/processed/asset_summary.csv`

### 5.2 风险层输入输出

#### 输入

* `returns_daily.csv`
* `configs/risk.yaml`

#### 输出

* `data/risk/cov_panel.parquet`
* `data/risk/corr_panel.parquet`
* `data/risk/vol_panel.csv`
* `data/risk/risk_estimation_meta.json`

### 5.3 模型层输入输出

#### 输入

* `cov_t`
* `corr_t`
* `sigma_t`
* `x_prev`
* `budget`
* `delta_band`
* `eta_smooth`
* `gamma_l2`
* `rho_penalty`
* `x_max`
* `mode`

#### 输出

* objective closure
* gradient closure
* Hessian closure
* `ctr`
* `ctb`
* `dr`
* `db`

### 5.4 solver 层输入输出

#### 输入

* `SolverState`
* `SolverOptions`

#### 输出

* `SolverResult`

  * `x_opt`
  * `objective_value`
  * `ctr`
  * `ctb`
  * `dr`
  * `db`
  * `obj_terms`
  * `converged`
  * `iterations_pg`
  * `iterations_newton`
  * `kkt_residual_final`
  * `constraint_violation`
  * `band_active`
  * `trace`

### 5.5 回测层输入输出

#### 输入

* `cov_panel.parquet`
* `corr_panel.parquet`
* `vol_panel.csv`
* `returns_daily.csv`
* `experiment.yaml`

#### 输出

* 每个策略的月度目标权重序列
* 每日 / 每月组合收益
* 换手时间序列
* 策略级 performance panel

### 5.6 导出层输入输出

#### 输入

* 回测层的策略结果对象
* solver diagnostics

#### 输出（固定）

* `summary_metrics.csv`
* `weights.csv`
* `ctr_long.csv`
* `ctb_long.csv`
* `dr_db_timeseries.csv`
* `objective_terms.csv`
* `turnover_timeseries.csv`
* `diagnostics.json`
* `analysis_pack.json`
* `run_manifest.json`

### 5.7 图表层输入输出

#### 输入

* `summary_metrics.csv`
* `weights.csv`
* `ctr_long.csv`
* `ctb_long.csv`
* `dr_db_timeseries.csv`
* `objective_terms.csv`
* `diagnostics.json`
* `analysis_pack.json`

#### 输出

* `fig_*.png`
* `fig_*.pdf`
* 对应底层 `csv`（如需要额外聚合）

---

## 6. 命名规范（冻结）

### 6.1 文件命名规范

统一使用：

* 小写英文；
* 下划线分隔；
* 禁止空格；
* 禁止中英文混杂文件名；
* 禁止临时文件名如 `final_final_v2.py`、`newtest.ipynb`。

示例：

* `returns_daily.csv`
* `risk_estimation_meta.json`
* `rb_ctb_band.py`
* `figure_dr_db_frontier.py`

### 6.2 模块命名规范

* objective 相关模块必须反映对象：`ctr.py`, `ctb.py`, `main_model.py`
* solver 相关模块必须反映算法角色：`pg.py`, `newton.py`, `project.py`
* backtest 模块必须反映流程：`engine.py`, `performance.py`, `turnover.py`

### 6.3 变量命名规范

遵守 `01_notation_master.md` 与 `02_definition_formula_ledger.md`：

* `x` / `weights`
* `cov`
* `corr`
* `sigma`
* `ctr`
* `ctb`
* `dr`
* `db`
* `delta_band`
* `eta_smooth`
* `gamma_l2`
* `rho_penalty`

禁止：

* 同一对象多种缩写并行；
* 用 `w`, `omega`, `weight_vec` 混指主权重；
* 用 `risk_balance`, `rb_loss`, `risk_gap` 等未注册名字代替 `dr`。

### 6.4 策略命名规范

固定策略代码：

* `EW`
* `GMV`
* `ERC`
* `CTB_ONLY`
* `RB_CTB_BAND`

正式结果文件中禁止出现：

* `main`
* `new_model`
* `model3`
* `best_strategy`
  作为策略名。

### 6.5 图表命名规范

统一采用：

* `fig_01_...`
* `fig_02_...`
* `table_01_...`

示例：

* `fig_01_capital_vs_risk.png`
* `fig_02_dr_db_frontier.pdf`
* `table_01_summary_metrics.tex`

---

## 7. 配置文件规范（冻结）

### 7.1 配置集中管理原则

所有可变设定必须进入 `configs/`，不允许在脚本内部硬编码。
代码中若出现配置常量，必须来自配置文件或总控契约。

### 7.2 建议配置文件

#### `configs/universe.yaml`

职责：

* ETF 列表
* 资产类别映射
* 上市筛选规则

#### `configs/data.yaml`

职责：

* 数据频率
* 价格字段选择
* 缺失值规则
* 调仓日历规则

#### `configs/risk.yaml`

职责：

* 协方差估计器
* 滚动窗口长度
* shrinkage 设定
* 风险估计版本号

#### `configs/model.yaml`

职责：

* 固定参数：`budget`, `x_max`
* 主模型选择：`mode`
* 是否启用 `gamma_l2`
* 是否启用 `rho_penalty`

#### `configs/solver.yaml`

职责：

* PG / Newton 参数
* 收敛阈值
* damping 规则
* fallback 规则

#### `configs/experiment.yaml`

职责：

* 样本切分
* 要运行的策略列表
* 是否做稳健性测试
* 输出目录设置

#### `configs/export.yaml`

职责：

* 要导出的结果文件列表
* 图表分辨率
* 表格格式

### 7.3 配置文件版本管理

每次正式 run 必须将实际使用的配置快照写入：

* `results/runs/<run_id>/config_snapshot/`

不得出现“结果出来了但不知道用了哪套参数”的情况。

---

## 8. run_id 与结果目录规范

### 8.1 run_id 规则

每次正式实验必须生成唯一 `run_id`，建议格式：

```text
YYYYMMDD-HHMM-<short_hash>
```

示例：

```text
20260418-1530-a1b2c3d4
```

### 8.2 单次 run 的目录结构

每次正式实验目录：

* 数据快照引用
* 配置快照
* 结构化结果
* 图表
* 表格
* 日志
* 诊断文件

### 8.3 结果目录职责

结果目录必须是**只读历史记录**：

* 不允许在已有 run 上追加覆盖式修改；
* 若修正代码后重跑，必须生成新的 `run_id`。

---

## 9. 日志、测试与质量控制

### 9.1 日志规范

必须区分三类日志：

#### 数据日志

* 数据下载
* 缺失处理
* 复权构造
* universe 可用性审计

#### 求解器日志

* PG/Newton 迭代信息
* fallback 触发
* 约束活跃状态

#### 运行日志

* run 开始/结束时间
* 使用配置版本
* 策略列表
* 关键异常

### 9.2 测试规范

建议测试层分三类：

#### 单元测试

* CtR / CtB 计算是否正确
* `project_to_capped_simplex()` 是否可行
* `dr` / `db` 计算与定义一致

#### 集成测试

* 单个调仓日期能否完整跑通
* 单个策略的结果导出是否完整

#### 回归测试

* 相同配置、相同数据下，关键 summary 指标应稳定在阈值内

### 9.3 数据审计检查

每次正式 run 前必须执行：

* universe completeness check
* return continuity check
* covariance PSD check
* calendar alignment check

---

## 10. 论文资产层规范

### 10.1 图表生成规则

论文图必须只从标准结果文件中生成。
不得：

* 在图表脚本中重跑模型；
* 在图表脚本中重估协方差；
* 在图表脚本中临时更改参数。

### 10.2 表格生成规则

论文表必须只从：

* `summary_metrics.csv`
* `weights.csv`
* `ctr_long.csv`
* `ctb_long.csv`
* `dr_db_timeseries.csv`
* `diagnostics.json`

等标准文件生成。

### 10.3 GPT 分析友好规则

为了让 GPT 后续可靠读取结果，每个图表必须满足：

1. 图有底层数据表；
2. 指标字段名与 `02_definition_formula_ledger.md` 完全一致；
3. 若图经过额外聚合，则聚合后的底层表也要保存。

---

## 11. 与总控文件的映射

### 11.1 与 `05_data_contract.md`

本仓库的数据层必须完全服从数据契约：

* universe 不可私自改；
* 收益率口径不可漂移；
* 风险估计主方案不可私自换。

### 11.2 与 `06_model_contract.md`

本仓库的 objective 层只允许：

* CtR-only
* CtB-only
* RB_CTB_BAND

禁止未注册第四模型。

### 11.3 与 `07_solver_contract.md`

solver 层必须服从统一接口、两阶段算法、诊断输出与 fallback 规则。

---

## 12. 冻结结论

从本文件起，本项目的代码仓固定为：

1. 研究管线型仓库，而不是杂糅脚本集合；
2. 按层分为 control / data / risk / objectives / solvers / backtest / exports / paper assets；
3. 每层职责唯一，不越权；
4. 所有配置集中在 `configs/`；
5. 每次 run 只生成一个 `run_id`，但必须同时输出论文资产和 GPT 可读资产；
6. 图表与表格一律从标准结果文件生成；
7. 不允许 Notebook 成为正式主线；
8. 不允许临时变量命名、临时模型命名、临时输出口径。

---

## 13. 项目目录树（固定，放在最后）

```text
project/
│
├── control/
│   ├── 00_project_charter.md
│   ├── 01_notation_master.md
│   ├── 02_definition_formula_ledger.md
│   ├── 03_thesis_skeleton.md
│   ├── 04_literature_map.md
│   ├── 05_data_contract.md
│   ├── 06_model_contract.md
│   ├── 07_solver_contract.md
│   ├── 08_repo_architecture.md
│   └── 09_writing_contract.md
│
├── configs/
│   ├── universe.yaml
│   ├── data.yaml
│   ├── risk.yaml
│   ├── model.yaml
│   ├── solver.yaml
│   ├── experiment.yaml
│   └── export.yaml
│
├── data/
│   ├── raw/
│   │   ├── prices_daily_raw.parquet
│   │   ├── dividends_raw.parquet
│   │   ├── splits_raw.parquet
│   │   └── metadata_raw.json
│   │
│   ├── processed/
│   │   ├── prices_daily_adj.parquet
│   │   ├── returns_daily.csv
│   │   ├── calendar_master.csv
│   │   ├── universe_mask.csv
│   │   ├── data_audit_report.json
│   │   └── asset_summary.csv
│   │
│   └── risk/
│       ├── cov_panel.parquet
│       ├── corr_panel.parquet
│       ├── vol_panel.csv
│       └── risk_estimation_meta.json
│
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── adjust.py
│   │   ├── calendar.py
│   │   ├── returns.py
│   │   └── audit.py
│   │
│   ├── risk/
│   │   ├── covariance.py
│   │   ├── correlation.py
│   │   ├── volatility.py
│   │   └── meta.py
│   │
│   ├── objectives/
│   │   ├── ctr.py
│   │   ├── ctb.py
│   │   ├── metrics.py
│   │   └── main_model.py
│   │
│   ├── solvers/
│   │   ├── core.py
│   │   ├── project.py
│   │   ├── pg.py
│   │   ├── newton.py
│   │   ├── active_set.py
│   │   └── diagnostics.py
│   │
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── rebalance.py
│   │   ├── performance.py
│   │   ├── turnover.py
│   │   └── strategy_registry.py
│   │
│   ├── exports/
│   │   ├── summary.py
│   │   ├── panels.py
│   │   ├── diagnostics_export.py
│   │   └── manifest.py
│   │
│   ├── figures/
│   │   ├── fig_capital_vs_risk.py
│   │   ├── fig_dr_db_frontier.py
│   │   ├── fig_ctr_heatmap.py
│   │   ├── fig_ctb_heatmap.py
│   │   ├── fig_corr_heatmap.py
│   │   ├── fig_calibration_heatmap.py
│   │   └── fig_solver_convergence.py
│   │
│   ├── tables/
│   │   ├── table_summary_metrics.py
│   │   ├── table_mechanism_metrics.py
│   │   └── table_robustness.py
│   │
│   └── utils/
│       ├── io.py
│       ├── logging_utils.py
│       ├── validators.py
│       ├── dates.py
│       └── registry.py
│
├── scripts/
│   ├── build_data.py
│   ├── estimate_risk.py
│   ├── run_backtest.py
│   ├── export_results.py
│   ├── build_figures.py
│   ├── build_tables.py
│   └── run_pipeline.py
│
├── results/
│   └── runs/
│       └── <run_id>/
│           ├── config_snapshot/
│           │   ├── universe.yaml
│           │   ├── data.yaml
│           │   ├── risk.yaml
│           │   ├── model.yaml
│           │   ├── solver.yaml
│           │   ├── experiment.yaml
│           │   └── export.yaml
│           │
│           ├── panels/
│           │   ├── weights.csv
│           │   ├── ctr_long.csv
│           │   ├── ctb_long.csv
│           │   ├── dr_db_timeseries.csv
│           │   ├── objective_terms.csv
│           │   └── turnover_timeseries.csv
│           │
│           ├── summaries/
│           │   ├── summary_metrics.csv
│           │   ├── analysis_pack.json
│           │   ├── diagnostics.json
│           │   └── run_manifest.json
│           │
│           ├── figures/
│           │   ├── fig_01_capital_vs_risk.png
│           │   ├── fig_01_capital_vs_risk.pdf
│           │   ├── fig_02_dr_db_frontier.png
│           │   ├── fig_03_ctr_heatmap.png
│           │   ├── fig_04_ctb_heatmap.png
│           │   ├── fig_05_corr_heatmap.png
│           │   ├── fig_06_calibration_heatmap.png
│           │   └── fig_07_solver_convergence.png
│           │
│           ├── tables/
│           │   ├── table_01_summary_metrics.csv
│           │   ├── table_01_summary_metrics.tex
│           │   ├── table_02_mechanism_metrics.csv
│           │   ├── table_02_mechanism_metrics.tex
│           │   └── table_03_robustness.csv
│           │
│           └── logs/
│               ├── pipeline.log
│               ├── solver.log
│               └── data_audit.log
│
├── notebooks/
│   ├── exploratory/
│   └── presentation/
│
├── paper/
│   ├── manuscript/
│   ├── figures/
│   ├── tables/
│   ├── appendix/
│   └── slides/
│
├── tests/
│   ├── test_data.py
│   ├── test_objectives.py
│   ├── test_projection.py
│   ├── test_solver.py
│   ├── test_backtest.py
│   └── test_exports.py
│
├── README.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```
