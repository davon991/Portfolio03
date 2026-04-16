# 08_result_contract.md

## 1. 文件使命

本文件冻结本项目的**结果输出契约**。
从本文件起，后续所有实验运行、回测输出、论文图表生成、表格生成、GPT 二次分析、答辩材料制作，都必须服从本文件规定的：

* 结果目录层级；
* 必须输出的标准文件；
* 各文件字段名；
* 图表与底层 CSV 的一一对应关系；
* 表格与结果目录的自动生成关系；
* GPT 可读 JSON / CSV 诊断资产；
* 命名规则与版本追踪规则。

本文件的目标是确保：

1. **每张图都有对应 CSV**；
2. **每张表都能从结果目录自动生成**；
3. **论文图表与 GPT 读取分析使用同一批 run 结果**；
4. **任何结论都能回溯到标准化输出文件**；
5. **不会出现“论文数字、图表数字、JSON 数字三套口径”的情况**。

---

## 2. 总原则（冻结）

### 2.1 单次 run、单一真相源原则

每次正式实验只允许有一个唯一 `run_id`。
该 `run_id` 下的结果目录是本次实验的唯一真相源（single source of truth）。

从该目录出发，必须同时生成：

* 论文图；
* 论文表；
* GPT 可读 CSV / JSON；
* 运行清单与诊断文件。

禁止：

* 从不同 run 混拼图表；
* 论文图来自 A run，而 summary 表来自 B run；
* 图表脚本临时重算新指标但不落盘。

### 2.2 结构化优先原则

图像文件本身不能成为唯一输出。
任何图表所展示的指标，都必须先以结构化文件形式存在：

* `csv`
* `json`
* `parquet`（若需要）

然后图表脚本只能从这些结构化文件中读入并绘图。

### 2.3 论文资产与分析资产并行原则

每次 run 必须同时输出两类结果：

#### A. 论文资产（paper assets）

* 论文图表
* 论文表格
* LaTeX 表格
* 可直接插入论文的 PNG / PDF 文件

#### B. GPT 分析资产（analysis assets）

* summary CSV
* long-format panel CSV
* solver diagnostics JSON
* analysis pack JSON
* objective term time series
* event / warning / fallback 记录

两类资产必须来自同一 run_id，且指标口径完全一致。

### 2.4 标准字段不可漂移原则

所有字段名必须服从：

* `01_notation_master.md`
* `02_definition_formula_ledger.md`
* `06_model_contract.md`

禁止：

* 同一对象多字段并行；
* 结果里同时出现 `db`, `ctb_var`, `corr_dispersion` 指向同一对象；
* `dr` / `db` 在不同文件里口径不同。

### 2.5 图表只读原则

论文图表脚本和答辩图表脚本只允许读取结果文件，不允许：

* 重新求解模型；
* 重新跑回测；
* 重新估计协方差；
* 自行修复数据。

---

## 3. 结果目录规范（冻结）

每次正式实验的结果目录固定为：

```text
results/runs/<run_id>/
```

其中 `<run_id>` 建议格式：

```text
YYYYMMDD-HHMM-<short_hash>
```

例如：

```text
20260420-1430-a1b2c3d4
```

该目录下必须固定包含以下子目录：

* `config_snapshot/`
* `panels/`
* `summaries/`
* `figures/`
* `tables/`
* `logs/`

禁止：

* 将关键输出散落在根目录；
* 将某类文件只存本地 notebook，不写入 run 目录；
* 对已完成 run 进行覆盖式修改。

---

## 4. 必须输出的结果文件（冻结）

## 4.1 配置快照层 `config_snapshot/`

### 使命

记录本次 run 的全部配置状态，确保可追溯性。

### 必须输出

* `universe.yaml`
* `data.yaml`
* `risk.yaml`
* `model.yaml`
* `solver.yaml`
* `experiment.yaml`
* `export.yaml`

### 作用

任何一轮实验的结果都必须能追溯到它使用了哪套配置。

---

## 4.2 面板层 `panels/`

这是所有图表和表格的**底层主数据层**。

### 4.2.1 `weights.csv`

#### 使命

记录所有策略在所有调仓日期下的权重配置。

#### 必须字段

* `date`
* `strategy`
* `asset`
* `weight`

#### 用途

* 权重热力图
* capital allocation bar chart
* 权重路径图
* 表格中的平均权重/代表时点权重

---

### 4.2.2 `ctr_long.csv`

#### 使命

记录所有策略在所有日期、所有资产上的 CtR。

#### 必须字段

* `date`
* `strategy`
* `asset`
* `ctr`

#### 用途

* CtR 热力图
* 风险贡献镜像条形图
* CtR 机制分析
* `dr` 的复核

---

### 4.2.3 `ctb_long.csv`

#### 使命

记录所有策略在所有日期、所有资产上的 CtB。

#### 必须字段

* `date`
* `strategy`
* `asset`
* `ctb`

#### 用途

* CtB 热力图
* CtB 横截面比较
* `db` 的复核
* 结构暴露分析

---

### 4.2.4 `dr_db_timeseries.csv`

#### 使命

记录每个策略在每个日期的 CtR 偏离度和 CtB 离散度。

#### 必须字段

* `date`
* `strategy`
* `dr`
* `db`
* `band_active`

#### 可选增强字段

* `db_margin`
* `dr_rank`
* `db_rank`

#### 用途

* rolling DR / DB 双面板
* active band 阴影图
* 机制对比图
* frontiers 的底层数据来源之一

---

### 4.2.5 `objective_terms.csv`

#### 使命

记录主模型在每个日期上的目标函数分解，便于解释模型是如何起作用的。

#### 必须字段

* `date`
* `strategy`
* `obj_total`
* `dr_term`
* `smooth_term`
* `l2_term`
* `band_penalty`

#### 用途

* 目标函数分解图
* 主模型机制解释
* `analysis_pack.json` 生成
* 论文正文说明“约束何时起作用”

---

### 4.2.6 `turnover_timeseries.csv`

#### 使命

记录每个策略每次调仓的换手。

#### 必须字段

* `date`
* `strategy`
* `turnover`

#### 可选增强字段

* `gross_turnover`
* `net_turnover`

#### 用途

* 换手时序图
* 参数 η 的解释
* summary metrics 计算

---

### 4.2.7 `perf_daily.csv`

#### 使命

记录各策略的日度组合收益与累计净值路径。

#### 必须字段

* `date`
* `strategy`
* `portfolio_return`
* `nav`
* `drawdown`

#### 用途

* 累计净值图
* 回撤图
* 年化绩效指标计算

---

### 4.2.8 `corr_structure_panel.csv`（可选但强烈建议）

#### 使命

记录代表窗口下的相关结构摘要，用于 CtB 和结构控制图表。

#### 推荐字段

* `date`
* `asset_i`
* `asset_j`
* `corr`
* `cluster_order_i`
* `cluster_order_j`

#### 用途

* 原始相关矩阵热图
* 重排/聚类相关矩阵热图

---

## 4.3 汇总层 `summaries/`

### 4.3.1 `summary_metrics.csv`

#### 使命

记录各策略的总体绩效与机制指标汇总，是论文所有 summary 表格的主来源。

#### 必须字段

* `strategy`
* `ann_return`
* `ann_vol`
* `sharpe`
* `max_drawdown`
* `turnover_mean`
* `turnover_p95`
* `dr_mean`
* `db_mean`
* `active_rate`

#### 可选增强字段

* `calmar`
* `sortino`
* `hit_ratio`
* `net_excess_return`

#### 用途

* 论文总体绩效表
* 摘要级答辩表格
* `analysis_pack.json` 摘要

---

### 4.3.2 `analysis_pack.json`

#### 使命

为 GPT 和后续自动分析准备的一份“高层摘要文件”。

#### 必须字段

* `run_id`
* `best_strategy`
* `strategy_order`
* `key_findings`
* `metric_deltas_vs_erc`
* `db_reduction_vs_erc`
* `dr_change_vs_erc`
* `active_rate`
* `recommended_figures`
* `warnings`

#### 推荐字段

* `solver_health_summary`
* `data_quality_flags`
* `robustness_highlights`

#### 用途

* GPT 后续机制分析
* 答辩摘要生成
* 论文结论草稿生成辅助

---

### 4.3.3 `diagnostics.json`

#### 使命

记录整个 run 的 solver、约束、回退、异常与健康状态。

#### 必须字段

* `run_id`
* `solve_success_rate`
* `fallback_rate`
* `active_rate`
* `kkt_residual_stats`
* `constraint_violation_stats`
* `band_violation_stats`
* `step_rejection_stats`
* `warning_dates`
* `failed_dates`

#### 用途

* 论文第 6 章 solver 诊断
* GPT 可读健康报告
* 稳健性与异常日期分析

---

### 4.3.4 `run_manifest.json`

#### 使命

记录一次 run 的元信息与文件清单。

#### 必须字段

* `run_id`
* `timestamp_start`
* `timestamp_end`
* `git_commit`（若可用）
* `config_hash`
* `strategies_run`
* `data_snapshot_ref`
* `output_files`
* `notes`

#### 用途

* 可追溯性
* 审计
* 复现

---

## 4.4 图表层 `figures/`

图表不是原始结果，而是基于标准 CSV/JSON 导出的**论文资产**。
但每张图必须有对应底层 CSV 作为唯一输入来源。

### 必须图表清单

#### 图 1 `fig_01_capital_vs_risk`

* 文件：

  * `fig_01_capital_vs_risk.png`
  * `fig_01_capital_vs_risk.pdf`
* 对应底层 CSV：

  * `weights.csv`
  * `ctr_long.csv`
* 作用：资本配置 vs 风险配置镜像条形图

#### 图 2 `fig_02_dr_db_frontier`

* 文件：

  * `fig_02_dr_db_frontier.png`
  * `fig_02_dr_db_frontier.pdf`
* 对应底层 CSV：

  * `summary_metrics.csv`
  * 或单独导出的 `dr_db_frontier.csv`
* 作用：CtR–CtB 双机制前沿图

#### 图 3 `fig_03_cumulative_nav`

* 文件：

  * `fig_03_cumulative_nav.png`
  * `fig_03_cumulative_nav.pdf`
* 对应底层 CSV：

  * `perf_daily.csv`
* 作用：累计净值图

#### 图 4 `fig_04_drawdown`

* 文件：

  * `fig_04_drawdown.png`
  * `fig_04_drawdown.pdf`
* 对应底层 CSV：

  * `perf_daily.csv`
* 作用：回撤图

#### 图 5 `fig_05_ctr_heatmap`

* 文件：

  * `fig_05_ctr_heatmap.png`
  * `fig_05_ctr_heatmap.pdf`
* 对应底层 CSV：

  * `ctr_long.csv`
* 作用：CtR 时变热力图

#### 图 6 `fig_06_ctb_heatmap`

* 文件：

  * `fig_06_ctb_heatmap.png`
  * `fig_06_ctb_heatmap.pdf`
* 对应底层 CSV：

  * `ctb_long.csv`
* 作用：CtB 时变热力图

#### 图 7 `fig_07_dr_db_timeseries`

* 文件：

  * `fig_07_dr_db_timeseries.png`
  * `fig_07_dr_db_timeseries.pdf`
* 对应底层 CSV：

  * `dr_db_timeseries.csv`
* 作用：rolling DR / DB 双面板 + active shading

#### 图 8 `fig_08_corr_heatmap`

* 文件：

  * `fig_08_corr_heatmap.png`
  * `fig_08_corr_heatmap.pdf`
* 对应底层 CSV：

  * `corr_structure_panel.csv`
  * 或 `corr_panel.parquet` 转出的标准切片 CSV
* 作用：原始/重排相关矩阵热图

#### 图 9 `fig_09_calibration_heatmap`

* 文件：

  * `fig_09_calibration_heatmap.png`
  * `fig_09_calibration_heatmap.pdf`
* 对应底层 CSV：

  * `calibration_log.csv`
* 作用：参数校准可行域热力图

#### 图 10 `fig_10_solver_convergence`

* 文件：

  * `fig_10_solver_convergence.png`
  * `fig_10_solver_convergence.pdf`
* 对应底层 CSV：

  * `solver_trace.csv`
  * 或 `diagnostics.json` + 单独 trace 文件
* 作用：PG / Newton 收敛图

### 图表强制规则

1. 每张图必须有明确底层 CSV；
2. 图表脚本只能读标准 CSV/JSON，不得重算核心指标；
3. 若某图需要额外聚合，必须先把聚合结果保存为新的 CSV，再画图；
4. 图文件必须同时输出 PNG 和 PDF 两种格式。

---

## 4.5 表格层 `tables/`

### 4.5.1 `table_01_summary_metrics`

#### 文件

* `table_01_summary_metrics.csv`
* `table_01_summary_metrics.tex`

#### 数据来源

* `summary_metrics.csv`

#### 作用

* 论文总体绩效比较表
* 答辩摘要主表

---

### 4.5.2 `table_02_mechanism_metrics`

#### 文件

* `table_02_mechanism_metrics.csv`
* `table_02_mechanism_metrics.tex`

#### 数据来源

* `summary_metrics.csv`
* `dr_db_timeseries.csv`
* `diagnostics.json`

#### 建议字段

* `strategy`
* `dr_mean`
* `db_mean`
* `active_rate`
* `turnover_mean`
* `turnover_p95`

#### 作用

* 论文机制比较表

---

### 4.5.3 `table_03_robustness`

#### 文件

* `table_03_robustness.csv`
* `table_03_robustness.tex`

#### 数据来源

* 不同 run 的 `summary_metrics.csv`
* 或同一 run 的 robustness panel

#### 作用

* 稳健性分析表

### 表格强制规则

1. 每张表都必须能从结果目录自动生成；
2. `.tex` 版本必须由对应 `.csv` 自动导出；
3. 表格脚本不得直接从运行时内存对象抓数据。

---

## 5. GPT 可读分析资产（冻结）

以下文件属于**强制输出**，不是可选项。

### 5.1 `analysis_pack.json`

高层摘要，供 GPT 快速读取整轮 run 的核心结论。

### 5.2 `diagnostics.json`

solver 与约束健康状况。

### 5.3 `summary_metrics.csv`

各策略 summary 指标表。

### 5.4 `weights.csv`

长表权重数据。

### 5.5 `ctr_long.csv`

CtR 长表。

### 5.6 `ctb_long.csv`

CtB 长表。

### 5.7 `dr_db_timeseries.csv`

机制时序。

### 5.8 `objective_terms.csv`

目标函数分解。

### 5.9 `turnover_timeseries.csv`

换手时序。

### 5.10 `run_manifest.json`

结果清单与元数据。

### 5.11 `solver_trace.csv`（强烈建议）

即使不是每次论文都展示，也建议强制输出，供 GPT 分析和 debug。

#### 推荐字段

* `date`
* `strategy`
* `phase` (`pg` / `newton`)
* `iter`
* `objective_value`
* `grad_norm`
* `kkt_residual`
* `constraint_violation`
* `band_active`
* `step_size`
* `damping`

---

## 6. 字段命名规范（冻结）

所有输出字段名必须与 `02_definition_formula_ledger.md` 一致。
以下字段为保留字段，禁止修改：

### 核心对象字段

* `weight`
* `ctr`
* `ctb`
* `dr`
* `db`

### 参数字段

* `delta_band`
* `eta_smooth`
* `gamma_l2`
* `rho_penalty`

### 诊断字段

* `band_active`
* `objective_value`
* `kkt_residual`
* `constraint_violation`
* `converged`
* `solver_status`

### 绩效字段

* `ann_return`
* `ann_vol`
* `sharpe`
* `max_drawdown`
* `turnover_mean`
* `turnover_p95`

禁止同义替代：

* `risk_contribution` 替代 `ctr`
* `corr_dispersion` 替代 `db`
* `objective` 替代 `obj_total`

若需要额外字段，必须在台账中登记并保持唯一口径。

---

## 7. 自动生成规则（冻结）

### 7.1 图表自动生成规则

图表脚本必须从 `results/runs/<run_id>/` 自动扫描并生成：

* PNG
* PDF
* 若需要，中间聚合 CSV

### 7.2 表格自动生成规则

表格脚本必须从结果目录自动读取标准文件并生成：

* CSV 表
* LaTeX 表

### 7.3 analysis asset 自动生成规则

`analysis_pack.json`、`diagnostics.json`、`run_manifest.json` 必须在 run 结束时自动生成。
不允许事后手工补写。

### 7.4 缺文件即视为 run 不完整

若一个正式 run 缺失以下任一核心文件，则该 run 视为不合格：

* `summary_metrics.csv`
* `weights.csv`
* `ctr_long.csv`
* `ctb_long.csv`
* `dr_db_timeseries.csv`
* `objective_terms.csv`
* `diagnostics.json`
* `analysis_pack.json`
* `run_manifest.json`

---

## 8. 与论文章节的映射

### 第 3 章 CtR 理论与可操作化

依赖：

* `ctr_long.csv`
* `weights.csv`
* `fig_01_capital_vs_risk`
* `fig_05_ctr_heatmap`

### 第 4 章 CtB 理论与可操作化

依赖：

* `ctb_long.csv`
* `dr_db_timeseries.csv`
* `fig_06_ctb_heatmap`
* `fig_08_corr_heatmap`

### 第 5 章 CtR–CtB 关系与协调模型

依赖：

* `summary_metrics.csv`
* `dr_db_timeseries.csv`
* `fig_02_dr_db_frontier`

### 第 6 章 统一求解器

依赖：

* `diagnostics.json`
* `solver_trace.csv`
* `fig_10_solver_convergence`

### 第 7 章 数据、参数校准与实验设计

依赖：

* `calibration_log.csv`
* `fig_09_calibration_heatmap`
* `run_manifest.json`

### 第 8 章 实证结果与机制分析

依赖：

* 所有 summary / panel / figure / table 文件

---

## 9. 与代码仓架构的映射

### `src/exports/`

负责生成：

* 所有 panel CSV
* 所有 summary CSV / JSON
* manifest
* diagnostics

### `src/figures/`

负责从 panel / summary 文件生成图

### `src/tables/`

负责从 panel / summary 文件生成表

### `scripts/export_results.py`

负责确保一个 run 结束后，所有标准输出文件齐全。

---

## 10. 质量控制与审计规则

### 10.1 结果完整性检查

每次 run 完成后必须自动检查：

* 标准文件是否齐全；
* 字段是否缺失；
* 各文件中的策略集合是否一致；
* 日期范围是否一致。

### 10.2 一致性检查

必须自动检查：

* `summary_metrics.csv` 是否可由 `perf_daily.csv` 重建；
* `dr_mean`, `db_mean` 是否与 `dr_db_timeseries.csv` 一致；
* `ctr` 与 `weights` 是否能复核资本 vs 风险图；
* `band_active` 是否与 `db > delta_band` 一致。

### 10.3 失败记录

若导出失败或图表缺失，必须在：

* `run_manifest.json`
* `analysis_pack.json`
* 日志文件

中记录。

---

## 11. 冻结结论

从本文件起，本项目的结果输出层固定如下：

1. 每次 run 必须在 `results/runs/<run_id>/` 下输出完整标准结果；
2. 每张图必须有对应 CSV；
3. 每张表必须能从结果目录自动生成；
4. 必须同时输出论文资产与 GPT 可读分析资产；
5. 所有字段名固定，不允许漂移；
6. 图表与表格脚本只能读取标准结果文件，不得重新求解模型或重算核心指标；
7. 缺失核心文件的 run 视为无效 run；
8. 后续论文与答辩只能引用符合本契约的结果文件。
