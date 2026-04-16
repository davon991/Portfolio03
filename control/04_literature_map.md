# 04_literature_map.md

## 使用说明
本文件不是“按年代排列的综述”，而是面向当前论文目录的**章节级文献地图**。  
每篇文献都标注其最适合的作用：

- 【理论对象】
- 【建模选择】
- 【数值方法】
- 【图表借鉴】

并注明**最适合引用到哪一节**。

---

# 第 1 章 绪论

## 1.1 研究背景：组合优化为何会走向 CtR / CtB

### 核心文献
1. **Markowitz (1952), Portfolio Selection**  
   - 角色：【理论对象】现代均值—方差组合选择的起点。  
   - 最适合引用：**1.1 研究背景**、**2.2 Markowitz 均值—方差模型**。  
   - 用途：说明“总体风险最小化”框架的出发点。 :contentReference[oaicite:4]{index=4}

2. **Tobin (1958), Liquidity Preference as Behavior Towards Risk**  
   - 角色：【理论对象】将无风险资产引入均值—方差框架的经典来源。  
   - 最适合引用：**1.1 研究背景**、**2.2/2.3 基础扩展**。  
   - 用途：说明从纯风险资产配置走向现实配置问题的过渡。 :contentReference[oaicite:5]{index=5}

3. **Sharpe (1963), A Simplified Model for Portfolio Analysis**  
   - 角色：【理论对象】单指数模型的经典来源。  
   - 最适合引用：**1.1 研究背景**、**2.4 Sharpe 单指数模型**。  
   - 用途：说明协方差结构简化与实务可计算性。 :contentReference[oaicite:6]{index=6}

4. **Sharpe (1964), Capital Asset Prices**  
   - 角色：【理论对象】CAPM 经典来源。  
   - 最适合引用：**1.1 研究背景**。  
   - 用途：说明风险—收益均衡思想如何影响后续组合理论。 :contentReference[oaicite:7]{index=7}

## 1.2 研究问题定位：为什么不是只做 CtR，或只做 CtB
1. **riskParityPortfolio package manual / vignette family**  
   - 角色：【建模选择】明确风险平价的核心是“equalize or distribute the risk contributions”。  
   - 最适合引用：**1.2 研究问题**、**3.2 ERC / risk budgeting**。  
   - 用途：说明 CtR 主线是 risk parity 文献的中心对象。 :contentReference[oaicite:8]{index=8}

2. **MSCI, Build Risk Parity Portfolios with Correlation Risk Attribution**  
   - 角色：【建模选择】把风险贡献拆成 exposure × volatility × correlation。  
   - 最适合引用：**1.2 研究问题**、**5.3 为什么 CtR 主目标 + CtB 约束**。  
   - 用途：说明 CtB 更像“相关暴露通道”的控制对象。 :contentReference[oaicite:9]{index=9}

---

# 第 2 章 预备知识与统一记号

## 2.1 收益率、协方差矩阵、组合方差
1. **Markowitz (1952)**  
   - 角色：【理论对象】组合方差、均值—方差语言。  
   - 最适合引用：**2.1**、**2.2**。 :contentReference[oaicite:10]{index=10}

2. **Boyd et al. (2024), Markowitz Portfolio Construction at Seventy**  
   - 角色：【建模选择】现代视角下重新梳理 Markowitz 框架。  
   - 最适合引用：**2.1** 或 **2.2** 的现代回顾段落。  
   - 用途：让第 2 章不是只有经典文献，也有较新的总述。 :contentReference[oaicite:11]{index=11}

## 2.2 Markowitz 与 long-only 约束
1. **Markowitz (1952)**  
   - 角色：【理论对象】最小方差与有效前沿。  
   - 最适合引用：**2.2**。 :contentReference[oaicite:12]{index=12}

2. **riskParityPortfolio manual**  
   - 角色：【建模选择】说明在 long-only、预算约束下，vanilla risk parity 是标准且可实现的设定。  
   - 最适合引用：**2.3 无卖空约束与数值求解必要性**。 :contentReference[oaicite:13]{index=13}

## 2.3 Sharpe 基础
1. **Sharpe (1963)**  
   - 角色：【理论对象】单指数模型。  
   - 最适合引用：**2.4**。 :contentReference[oaicite:14]{index=14}

2. **Sharpe (1964)**  
   - 角色：【理论对象】CAPM 与市场均衡。  
   - 最适合引用：**2.4**。 :contentReference[oaicite:15]{index=15}

## 图表借鉴
1. **Palomar, Graph-Based Portfolios slides**  
   - 角色：【图表借鉴】相关矩阵热图、聚类/重排热图。  
   - 最适合引用/借鉴：**图 3：样本期相关矩阵热图**。 :contentReference[oaicite:16]{index=16}

---

# 第 3 章 CtR 的表征理论与 long-only 可操作化

## 3.1 CtR / ERC / Risk Budgeting 的核心理论对象
1. **Maillard, Roncalli, Teiletche, On the Properties of Equally-Weighted Risk Contributions Portfolios**  
   - 角色：【理论对象】ERC 的性质、分散化意义、风险平价核心文献。  
   - 最适合引用：**3.1 风险贡献定义**、**3.2 ERC / risk budgeting**。 :contentReference[oaicite:17]{index=17}

2. **Bruder & Roncalli, Managing Risk Exposures Using the Risk Budgeting Approach**  
   - 角色：【理论对象】【建模选择】risk budgeting 的系统性表述。  
   - 最适合引用：**3.2**、**5.3**。  
   - 用途：说明为什么 CtR 是预算型对象。 :contentReference[oaicite:18]{index=18}

3. **Roncalli, Introduction to Risk Parity and Budgeting**  
   - 角色：【理论对象】risk parity / budgeting 的系统教材型来源。  
   - 最适合引用：**3.1–3.4** 的背景和术语统一。 :contentReference[oaicite:19]{index=19}

## 3.2 数值方法：CtR 求解
1. **Spinu (2013), An Algorithm for Computing Risk Parity Weights**  
   - 角色：【数值方法】risk parity 凸化 / 快速算法的重要来源。  
   - 最适合引用：**3.5 CtR 数值求解思想**、**6.1/6.2 统一求解器背景**。  
   - 说明：web 结果主要通过后续算法文献间接确认其地位。 :contentReference[oaicite:20]{index=20}

2. **Griveau-Billion, Richard, Roncalli (2013), A Fast Algorithm for Computing High-dimensional Risk Parity Portfolios**  
   - 角色：【数值方法】CCD / Newton / SQP 比较。  
   - 最适合引用：**3.5**、**6.1**。 :contentReference[oaicite:21]{index=21}

3. **Bai, Scheinberg, Tütüncü (2016), Least-Squares Approach to Risk Parity**  
   - 角色：【数值方法】有约束时 risk parity 可能不存在，或需要替代 formulation。  
   - 最适合引用：**3.4**、**6.1**。  
   - 用途：支持“约束下问题不再是最简单情形”。 :contentReference[oaicite:22]{index=22}

## 3.3 long-only 可操作化
1. **riskParityPortfolio manual**  
   - 角色：【建模选择】支持把 CtR 偏离度作为 long-only 风险平价优化的直接对象。  
   - 最适合引用：**3.6 CtR 在 long-only 下的可操作化**。 :contentReference[oaicite:23]{index=23}

## 图表借鉴
1. **barplotPortfolioRisk documentation**  
   - 角色：【图表借鉴】资本配置 vs 风险配置镜像条形图。  
   - 最适合借鉴：**图 4：资本配置 vs 风险配置镜像条形图**。 :contentReference[oaicite:24]{index=24}

2. **S&P Indexing Risk Parity Strategies**  
   - 角色：【图表借鉴】权重与风险贡献对照图、不同资产风险贡献随配置变化的图。  
   - 最适合借鉴：**图 4**、**图 17**。 :contentReference[oaicite:25]{index=25}

---

# 第 4 章 CtB 的表征理论与 long-only 可操作化

## 4.1 CtB 的理论对象：equal-correlation / diversification 线
1. **Choueifaty & Coignard (2008), Toward Maximum Diversification**  
   - 角色：【理论对象】最大分散组合的经典来源；核心性质之一是所持资产与组合具有相同正相关。  
   - 最适合引用：**4.1 CtB 的定义与相关暴露含义**、**4.7 long-only 可操作化的外部支撑**。 :contentReference[oaicite:26]{index=26}

2. **Choueifaty, Froidure, Reynier (2013), Properties of the Most Diversified Portfolio**  
   - 角色：【理论对象】MDP 的数学性质、mean-variance 解释与 invariance properties。  
   - 最适合引用：**4.1**、**4.2**、**5.3**。 :contentReference[oaicite:27]{index=27}

3. **Jia et al. (2025), Robustness Characterizations of Continuous-Time Equal-Correlation Investment Strategy**  
   - 角色：【理论对象】【建模选择】equal-correlation 的近期 robustness 解释。  
   - 最适合引用：**4.1**、**4.7** 或 **9.3 展望**。  
   - 用途：给 CtB/equal-correlation 一条现代扩展线。 :contentReference[oaicite:28]{index=28}

## 4.2 CtB 与组合结构：balanced baskets / subset correlation
1. **Bailey & Lopez de Prado, Balanced Baskets**  
   - 角色：【理论对象】【建模选择】【数值方法】同时比较 ERC、maximum diversification、subset correlation，并给出 Python 代码与收敛图。  
   - 最适合引用：**4.2**、**4.3**、**4.4**、**6.5**。  
   - 这是你第 4 章和第 6 章之间非常关键的桥梁文献。 :contentReference[oaicite:29]{index=29}

## 4.3 long-only 下的 CtB 可操作化
1. **Choueifaty & Coignard (2008)**  
   - 角色：【建模选择】支持“回到 CtB 原始对象”，而不是坚持某个 signed surrogate。  
   - 最适合引用：**4.6–4.7**。 :contentReference[oaicite:30]{index=30}

2. **Choueifaty, Froidure, Reynier (2013)**  
   - 角色：【建模选择】支持 CtB / correlation structure 在 long-only diversified portfolio 中仍然有自然意义。  
   - 最适合引用：**4.7**。 :contentReference[oaicite:31]{index=31}

## 图表借鉴
1. **TOBAM materials derived from MDP line**  
   - 角色：【图表借鉴】因子暴露/相关暴露稳定性图。  
   - 最适合借鉴：**图 7/8/23**（CtB heatmap、相关矩阵结构图）。 :contentReference[oaicite:32]{index=32}

2. **Balanced Baskets**  
   - 角色：【图表借鉴】不同 balanced basket 方法的收敛图、机制比较图。  
   - 最适合借鉴：**图 9：CtB-only 收敛图**、**图 12：主模型收敛图**。 :contentReference[oaicite:33]{index=33}

---

# 第 5 章 CtR–CtB 关系、不可兼容性与协调模型选择

## 5.1 为什么不能双严格平价
1. **Maillard–Roncalli–Teiletche (ERC)** + **Choueifaty–Coignard / Choueifaty–Froidure–Reynier (MDP)**  
   - 角色：【理论对象】分别代表 CtR 平价和 CtB / equal-correlation 平价两条线。  
   - 最适合引用：**5.1 关系式后的讨论**、**5.2 一般情形下不能同时严格平价**。  
   - 用途：用两条文献线解释“两个平价对象的归一化不同”。 :contentReference[oaicite:34]{index=34}

2. **Balanced Baskets**  
   - 角色：【建模选择】直接把 ERC、Maximum Diversification、subset correlation 并列比较。  
   - 最适合引用：**5.2–5.3**。  
   - 用途：支持“协调而非强行统一”的论文路线。 :contentReference[oaicite:35]{index=35}

## 5.2 为什么是 “CtR 主目标 + CtB 约束”
1. **Bruder & Roncalli (Risk Budgeting Approach)**  
   - 角色：【建模选择】支持 CtR 作为 budget object / primary allocation engine。  
   - 最适合引用：**5.3 协调模型选择**。 :contentReference[oaicite:36]{index=36}

2. **MSCI Correlation Risk Attribution paper**  
   - 角色：【建模选择】支持 CtB 更自然地作为 correlation channel / structural guardrail。  
   - 最适合引用：**5.3**。  
   - 这篇文献是你回答“为什么不是 CtB 做主目标”的最好外部支撑。 :contentReference[oaicite:37]{index=37}

3. **Choueifaty / MDP line**  
   - 角色：【建模选择】支持 CtB 更像 diversification / correlation structure 对象。  
   - 最适合引用：**5.3**。 :contentReference[oaicite:38]{index=38}

## 图表借鉴
1. **S&P Indexing Risk Parity Strategies**  
   - 角色：【图表借鉴】权重—风险贡献对照、风险来源变化图。  
   - 最适合借鉴：**图 10：\(D_R\)–\(D_B\) 前沿图** 的讲述方式。 :contentReference[oaicite:39]{index=39}

2. **MSCI Correlation Risk Attribution**  
   - 角色：【图表借鉴】三维贡献拆分（exposure / volatility / correlation）思路。  
   - 最适合借鉴：**图 11：CtR/CtB 横截面对照图** 的解释。 :contentReference[oaicite:40]{index=40}

---

# 第 6 章 统一求解器与数值实现

## 6.1 求解器理论背景
1. **Griveau-Billion et al. (2013)**  
   - 角色：【数值方法】risk parity 的 CCD / Newton / SQP 对照。  
   - 最适合引用：**6.1–6.3**。 :contentReference[oaicite:41]{index=41}

2. **Spinu (2013) via robust algorithm references**  
   - 角色：【数值方法】risk parity 求解的凸化/高效算法背景。  
   - 最适合引用：**6.1**。 :contentReference[oaicite:42]{index=42}

3. **riskParityPortfolio manual**  
   - 角色：【数值方法】说明带 no-shortselling、budget constraints、box constraints 时仍可统一处理。  
   - 最适合引用：**6.1**、**6.5**。 :contentReference[oaicite:43]{index=43}

4. **Balanced Baskets**  
   - 角色：【数值方法】三类 balanced basket 方法的实现对照与收敛展示。  
   - 最适合引用：**6.5**。 :contentReference[oaicite:44]{index=44}

## 图表借鉴
1. **Balanced Baskets convergence plots**  
   - 角色：【图表借鉴】收敛曲线、权重路径图。  
   - 最适合借鉴：**图 12、图 13**。 :contentReference[oaicite:45]{index=45}

2. **riskParityPortfolio package examples**  
   - 角色：【图表借鉴】risk allocation barplots。  
   - 最适合借鉴：**图 12** 的对照呈现方式。 :contentReference[oaicite:46]{index=46}

---

# 第 7 章 数据、参数校准与实验设计

## 7.1 ETF universe / multi-asset design
1. **S&P Risk Parity Indices Methodology**  
   - 角色：【建模选择】multi-asset risk parity universe 的实务范式。  
   - 最适合引用：**7.1 ETF universe 与样本期**。  
   - 用途：说明多资产 risk parity 在股票、固定收益、商品之间配置是成熟实务。 :contentReference[oaicite:47]{index=47}

2. **S&P Indexing Risk Parity Strategies**  
   - 角色：【建模选择】ETF / index-based multi-asset risk parity 的实证风格。  
   - 最适合引用：**7.1–7.2**。 :contentReference[oaicite:48]{index=48}

## 7.2 风险估计与稳健性
1. **RiskMetrics technical documents / practical guide**  
   - 角色：【数值方法】波动率与相关矩阵估计的实务背景。  
   - 最适合引用：**7.2 风险输入与协方差估计**。 :contentReference[oaicite:49]{index=49}

## 7.3 参数不是拍脑袋
1. **Bruder & Roncalli / risk budgeting line**  
   - 角色：【建模选择】支持预算与约束分层。  
   - 最适合引用：**7.3 参数分层**。 :contentReference[oaicite:50]{index=50}

2. **你自己的 calibration protocol（内部文件）**  
   - 角色：【建模选择】正文应以内部 protocol 为主；外部文献只是给“多资产 risk parity 要做规则化参数选择”的背景。

## 图表借鉴
1. **Palomar slides / book**  
   - 角色：【图表借鉴】数据探索、相关矩阵热图、rolling statistics。  
   - 最适合借鉴：**图 15–16**。 :contentReference[oaicite:51]{index=51}

2. **S&P / MSCI methodology papers**  
   - 角色：【图表借鉴】multi-asset allocation tables、risk attribution panels。  
   - 最适合借鉴：**参数说明图表、ETF 分类表**。 :contentReference[oaicite:52]{index=52}

---

# 第 8 章 实证结果与机制分析

## 8.1 绩效与机制双证据
1. **S&P Indexing Risk Parity Strategies**  
   - 角色：【建模选择】【图表借鉴】risk contribution 与 portfolio weights 对照、multi-asset performance panels。  
   - 最适合引用：**8.1 总体绩效**、**8.2 CtR 机制结果**。 :contentReference[oaicite:53]{index=53}

2. **MSCI Correlation Risk Attribution**  
   - 角色：【建模选择】【图表借鉴】把风险来源拆成暴露、波动率、相关性。  
   - 最适合引用：**8.3 CtB 机制结果**、**8.4 协调效果**。 :contentReference[oaicite:54]{index=54}

3. **Balanced Baskets**  
   - 角色：【建模选择】【图表借鉴】多种 balanced basket 方法的机制比较。  
   - 最适合引用：**8.4 协调效果**、**8.5 稳健性**。 :contentReference[oaicite:55]{index=55}

## 图表借鉴（主文强推荐）
1. **barplotPortfolioRisk** → 资本配置 vs 风险配置镜像图  
   - 对应：**图 17**。 :contentReference[oaicite:56]{index=56}

2. **S&P risk parity research** → 风险来源变化、权重-风险贡献对照  
   - 对应：**图 17、图 18**。 :contentReference[oaicite:57]{index=57}

3. **Palomar heatmap / clustering slides** → 相关矩阵热图与重排热图  
   - 对应：**图 23**。 :contentReference[oaicite:58]{index=58}

4. **Balanced Baskets convergence and method comparison**  
   - 对应：**图 21–24** 中的机制图与收敛图。 :contentReference[oaicite:59]{index=59}

---

# 第 9 章 结论与展望

## 适合回收的文献
1. **Maillard–Roncalli–Teiletche / Bruder–Roncalli**  
   - 回收第 3 章主线：CtR 是预算型对象。 :contentReference[oaicite:60]{index=60}

2. **Choueifaty–Coignard / Choueifaty–Froidure–Reynier**  
   - 回收第 4 章主线：CtB 是相关暴露 / 分散化对象。 :contentReference[oaicite:61]{index=61}

3. **Jia et al. (2025)**  
   - 放在展望：equal-correlation 的 robustness 理论仍在扩展。 :contentReference[oaicite:62]{index=62}

4. **risk parity under heavy tails / recent robust RP references**  
   - 放在展望：未来可把本文框架扩展到重尾和时变波动环境。 :contentReference[oaicite:63]{index=63}

---

# 最后建议：文献使用优先级

## A 级（必须进正文）
- Markowitz (1952)  
- Sharpe (1963)  
- Maillard, Roncalli, Teiletche  
- Bruder & Roncalli  
- Choueifaty & Coignard (2008)  
- Choueifaty, Froidure, Reynier (2013)  
- Bailey & Lopez de Prado (Balanced Baskets)  
- MSCI correlation risk attribution paper  
- riskParityPortfolio manual / documentation

## B 级（强烈建议）
- Tobin (1958)  
- Griveau-Billion et al.  
- Spinu-related algorithm references  
- S&P risk parity research / methodology  
- Palomar graph portfolio slides

## C 级（展望或补充）
- Jia et al. (2025) equal-correlation robustness  
- heavy-tail risk parity references  
- 其他 recent equal-correlation preprints