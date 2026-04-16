# 05_data_contract.md

## 1. 文件使命

本文件冻结本项目的数据对象、ETF universe、数据字段、预处理规则、调仓频率、样本切分与风险估计方案。  
后续所有代码模块、实验、图表、论文正文和答辩材料，都必须遵守本文件的数据口径。

本文件服务以下目标：

1. 保证 CtR、CtB 和主模型都基于同一套 ETF 数据宇宙；
2. 保证 long-only 实证的输入口径固定；
3. 保证结果可复现；
4. 保证每张图、每张表都能追溯到明确的数据文件和处理规则。

---

## 2. 固定 ETF universe

### 2.1 主 universe（冻结版）

本项目默认使用以下 10 只大类资产 ETF：

#### 股票（Equity）
1. **SPY** — 美国大盘股票  
2. **VEA** — 发达市场股票（美国除外）  
3. **EEM** — 新兴市场股票  

#### 利率（Rates）
4. **IEF** — 美国中久期国债（7–10 年）  
5. **TLT** — 美国长久期国债（20+ 年）  

#### 通胀（Inflation）
6. **TIP** — 美国通胀保值国债（TIPS）  

#### 信用（Credit）
7. **LQD** — 美元投资级公司债  

#### 实物资产（Real Assets）
8. **VNQ** — 美国 REITs / 上市房地产  
9. **GLD** — 黄金  
10. **DBC** — 广义商品  

### 2.2 各 ETF 的角色说明

- **SPY**：美国核心权益 beta，代表美国大型股风险。  
- **VEA**：发达市场 ex-US 风险，避免把全球股票全部压缩成美国单一权益风险。  
- **EEM**：新兴市场风险，补足全球股票的结构差异。  
- **IEF**：中久期利率风险。  
- **TLT**：长久期利率风险，用于形成更丰富的利率 term-structure 暴露。  
- **TIP**：通胀保护风险。  
- **LQD**：信用利差风险。  
- **VNQ**：房地产权益风险。  
- **GLD**：贵金属/避险实物资产风险。  
- **DBC**：广义商品期货风险。  

### 2.3 选择理由

本 universe 的设计原则是：

1. **覆盖完整**：股票、利率、通胀、信用、实物资产五大类必须都有。  
2. **数量克制**：总数固定为 10，只覆盖最关键的大类风险源，不引入过多冗余 ETF。  
3. **可解释性强**：每只 ETF 都对应一个清晰的宏观风险袖口，便于 CtR / CtB 图表解释。  
4. **适合 long-only 论文**：不会因为资产过多而使 long-only 风险预算、CtB 离散度和相关矩阵解释失真。  
5. **适合答辩**：评委容易理解“股票—利率—通胀—信用—实物资产”这五层结构。

### 2.4 ETF 官方目标（用于论文写作时的外部说明）

SPY 的目标是跟踪 S&P 500；VEA 跟踪发达市场股票；EEM 提供新兴市场大中盘股票敞口；IEF 和 TLT 分别跟踪中久期和长久期美国国债；TIP 跟踪美国通胀保值国债；LQD 跟踪美元投资级公司债；VNQ 跟踪美国房地产指数；GLD 反映金价表现；DBC 跟踪分散化商品指数。 :contentReference[oaicite:1]{index=1}

---

## 3. 数据频率与研究时间单位

### 3.1 原始下载频率
原始行情统一按 **日频** 下载。

### 3.2 主研究频率
主研究频率为 **月度再平衡**。

### 3.3 设计原则
采用“日频数据估计风险，月频进行调仓”：

- 日频足以支持协方差矩阵、波动率、相关矩阵、CtB 的稳定估计；
- 月频调仓可以显著降低换手与噪声，使主模型中的平滑项 \(\eta\) 更有现实解释；
- 也更适合硕士论文的可解释性和答辩展示。

---

## 4. 数据源字段契约

### 4.1 原始下载字段（最低要求）
每只 ETF 必须至少下载以下字段：

- `date`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`
- `dividends`（若数据源直接支持）
- `splits`（若数据源直接支持）

### 4.2 主研究使用字段
主研究实际使用的价格字段优先级如下：

1. **`adj_close` 优先**  
2. 若数据源不提供可靠的 `adj_close`，则使用 `close + dividends + splits` 自行复原总收益价格  
3. 不允许直接用未复权 `close` 构造主回测收益率

### 4.3 数据文件层次

#### 原始层
- `data/raw/prices_daily_raw.parquet`
- `data/raw/dividends_raw.parquet`
- `data/raw/splits_raw.parquet`
- `data/raw/metadata_raw.json`

#### 清洗后层
- `data/processed/prices_daily_adj.parquet`
- `data/processed/returns_daily.csv`
- `data/processed/calendar_master.csv`
- `data/processed/universe_mask.csv`

---

## 5. 预处理规则（冻结）

### 5.1 交易日对齐
- 以全 universe 的交易日并集构造主交易日历；
- 每只 ETF 按该主日历对齐；
- 非交易日不补造收益率；
- 仅在价格对齐层做缺失标记，不在原始层做插值。

### 5.2 缺失值处理
- 若 ETF 在样本早期尚未上市，则该阶段记为“不可用”，不得向前填充；
- 若某日出现孤立缺失，允许在**价格层**以前一有效价格做前向填充，但必须在 metadata 中记录；
- 若连续缺失超过预设阈值（例如 5 个交易日），该区间视为异常窗口，需要在数据审计中记录。

### 5.3 复权原则
- 一律以总收益口径为主；
- 使用 `adj_close` 或等价重构总收益价格；
- 所有回测收益率、协方差估计、CtR/CtB 计算一律基于复权后的收益率。

### 5.4 异常值检查
- 对单日收益率做基础异常值审计；
- 若发现明显由数据错误导致的极端值，必须先回查原始数据再决定是否修正；
- 不允许无记录地对收益率进行 winsorize。

### 5.5 Universe 可用性掩码
生成 `universe_mask.csv`，记录每个交易日、每只 ETF 是否可用于风险估计与回测。  
字段：

- `date`
- `asset`
- `is_live`
- `is_tradeable`
- `has_full_history_window`

---

## 6. 收益率构造规则

### 6.1 日收益率
对于复权价格 \(P_{i,t}^{adj}\)，定义日收益率：

\[
r_{i,t}=\frac{P_{i,t}^{adj}}{P_{i,t-1}^{adj}}-1
\]

代码层建议同时保留：
- 简单收益率
- 对数收益率（仅用于稳健性分析或附录）

### 6.2 主线收益率口径
主线统一使用 **简单收益率** 进入：

- 协方差估计
- CtR
- CtB
- 回测累计收益

### 6.3 月收益率
月收益率由日复权收益率累积得到，不直接下载月度行情。

---

## 7. 样本起点与样本切分

### 7.1 主样本起点
主回测起点必须满足两个条件：

1. universe 中所有 ETF 都已可用；
2. 已经积累足够长的滚动风险估计窗口。

因此，主样本起点不预先写死为某个日期，而由：

\[
\text{sample\_start}=\max(\text{latest ETF live date})+\text{risk window}
\]

自动确定。

### 7.2 样本切分原则
采用时间顺序切分，禁止随机切分：

- `train`
- `validation`
- `test`

### 7.3 推荐切分方式
当样本长度足够时，推荐：

- 训练集：前 50%
- 验证集：中间 25%
- 测试集：后 25%

或采用按自然时间的三段切分，但必须保持：
- 参数校准只用 train / validation；
- test 只用于最终报告。

---

## 8. 调仓与回测时序规则

### 8.1 调仓频率
固定为 **月度调仓**。

### 8.2 调仓时点
- 在每个月最后一个交易日收盘后，使用截至该日的全部历史信息估计风险矩阵并生成下一期目标权重；
- 下一持有期为下个月全部交易日。

### 8.3 信息集原则
- 不允许使用未来数据；
- 所有权重计算严格基于调仓时点之前的信息。

### 8.4 权重持有规则
- 月初建仓后，当月不再日内调仓；
- 若无特殊说明，不做再平衡内插值。

### 8.5 换手定义
月度换手统一定义为：
\[
\mathrm{TO}_t=\frac12\sum_i |x_{i,t}^{new}-x_{i,t}^{old\_drifted}|
\]
其中 `old_drifted` 表示旧组合在月末经价格漂移后的权重。

---

## 9. 协方差估计方案（冻结主线）

### 9.1 主线方案
主线风险矩阵 \(V_t\) 采用：

- **滚动窗口日收益率**
- **月度更新**
- **Ledoit–Wolf shrinkage covariance**

### 9.2 推荐窗口长度
主线默认窗口：

- **756 个交易日**（约 3 年）

理由：

1. universe 只有 10 个资产，维度不高，但跨资产协方差具有明显时变性；
2. 3 年窗口比 1 年窗口更稳，不会让 CtB 结构过度噪声化；
3. 月度调仓下，3 年日频窗口能兼顾稳定性与响应性；
4. 适合论文中的机制图与热力图解释。

### 9.3 稳健性方案
稳健性检验允许额外比较：

- 504 日窗口（约 2 年）
- 1260 日窗口（约 5 年）
- 样本协方差
- EWMA 协方差（附录或稳健性）

但正文主线只保留一个主方案，避免口径漂移。

### 9.4 协方差估计输入
- 只使用 `returns_daily.csv`
- 只在 `has_full_history_window = 1` 的时点计算

### 9.5 数值稳定处理
若协方差矩阵近奇异：

- 首先检查数据问题；
- 若数据无问题，则允许施加极小 ridge regularization；
- regularization 必须记录在 metadata 中，不允许静默处理。

### 9.6 风险矩阵输出
每个调仓时点输出：

- `cov_t.parquet` 或合并写入 `cov_panel.parquet`
- `corr_t.parquet` 或合并写入 `corr_panel.parquet`
- `risk_estimation_meta.json`

S&P 的风险平价方法文档说明，多资产风险平价配置通常围绕 equity、fixed income、commodities 等大类风险敞口进行，且依赖长期历史波动率/风险估计；这和本文用大类资产 ETF、月频调仓、滚动风险估计的设计方向是一致的。:contentReference[oaicite:2]{index=2}

---

## 10. 与后续模型模块的衔接

### 10.1 传递给模型层的对象
在每个调仓时点 \(t\)，数据模块必须向模型层传递：

- `x_prev`：上一期漂移后权重
- `mu_t`：均值向量（若主模型需要；正文主线可不使用）
- `cov_t = V_t`
- `corr_t = C_t`
- `sigma_t`：资产波动率向量
- `returns_window_t`
- `asset_list`
- `date_t`

### 10.2 不允许模型层重复做的事
模型层不得：
- 再次自行清洗数据；
- 再次自行对齐交易日历；
- 自行改变收益率口径；
- 静默替换协方差估计器。

---

## 11. 数据模块的标准输出文件

### 11.1 原始层输出
- `prices_daily_raw.parquet`
- `dividends_raw.parquet`
- `splits_raw.parquet`
- `metadata_raw.json`

### 11.2 清洗层输出
- `prices_daily_adj.parquet`
- `returns_daily.csv`
- `calendar_master.csv`
- `universe_mask.csv`

### 11.3 风险层输出
- `cov_panel.parquet`
- `corr_panel.parquet`
- `vol_panel.csv`
- `risk_estimation_meta.json`

### 11.4 供 GPT 后续分析读取的摘要文件
- `data_audit_report.json`
- `missingness_report.csv`
- `asset_summary.csv`

字段建议：

#### `asset_summary.csv`
- `asset`
- `category`
- `first_valid_date`
- `last_valid_date`
- `obs_count`
- `mean_daily_return`
- `daily_vol`
- `max_drawdown_proxy`

#### `data_audit_report.json`
- `calendar_start`
- `calendar_end`
- `sample_start`
- `n_assets`
- `n_missing_blocks`
- `forward_fill_events`
- `assets_with_short_history`

---

## 12. 论文与答辩中对数据部分的标准表述

正文中关于数据应统一表述为：

> 本文以大类资产 ETF 为研究对象，构造覆盖股票、利率、通胀、信用和实物资产的 long-only 多资产配置 universe。  
> 使用日频复权价格构造收益率，并按月度频率实施调仓。  
> 风险矩阵采用滚动窗口日收益率的 Ledoit–Wolf 收缩协方差估计，每月更新一次。  
> 参数校准采用严格的时间顺序切分，测试集不参与调参。

---

## 13. 冻结结论

从本文件起，本项目的数据层冻结如下：

1. 默认 universe 为 10 只大类资产 ETF：SPY、VEA、EEM、IEF、TLT、TIP、LQD、VNQ、GLD、DBC；
2. 原始数据按日频下载，主研究按月频调仓；
3. 收益率统一采用复权价格构造的简单收益率；
4. 主风险矩阵采用 756 日滚动 Ledoit–Wolf shrinkage covariance；
5. 样本切分采用严格时间顺序 train / validation / test；
6. 数据层必须输出原始层、清洗层、风险层和摘要层文件；
7. 后续模型、代码、论文、图表不得绕开本文件自行更改数据口径。