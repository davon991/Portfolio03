# 03_thesis_skeleton.md

## 论文暂定题目
风险贡献与相关暴露协调下的 long-only 大类资产 ETF 投资组合优化：
CtR / CtB 理论、可操作化与统一求解

---

## 第 1 章 绪论

### 使命
给出研究背景、研究问题、论文贡献与整体技术路线。  
这一章不承担推导任务，只负责把全文研究对象“钉住”。

### 本章必须回答的问题
1. 为什么传统均值—方差框架不足以刻画“组合内部的平衡”？
2. 为什么需要同时研究 CtR 与 CtB？
3. 为什么本文的重点不是追求一般情形下的双严格平价，而是研究 long-only 下的可操作化与协调？
4. 为什么选择大类资产 ETF 作为数据对象？

### 核心内容
- 组合优化研究背景
- CtR 与 CtB 的问题意识
- long-only 投资世界下的现实约束
- 研究问题 RQ1–RQ5
- 论文结构说明
- 研究贡献

### 核心公式
本章不展开正式推导，只可提前出现符号性公式：
\[
\mathrm{CtR}_i(x)=\frac{x_i(Vx)_i}{x^\top Vx},\qquad
\mathrm{CtB}_i(x)=\frac{(Vx)_i}{\sigma_i\sigma_p(x)}
\]
用于引出问题。

### 图表
- 图 1：论文技术路线图（理论层 → 可操作化层 → 协调模型层 → 求解器 → 实证）
- 表 1：全文符号与章节对应关系简表

### 实验
无正式实验，只可预告 ETF universe 和主要对照组。

---

## 第 2 章 预备知识与统一记号

### 使命
建立全文统一的收益率—协方差矩阵—权重语言，并给出经典优化基础。  
这章是基础层，不是主创新章。

### 本章必须回答的问题
1. 资产收益、组合收益、协方差矩阵如何统一表示？
2. Markowitz 与 Sharpe 在本文中扮演什么角色？
3. long-only 约束如何进入优化问题？

### 核心内容
#### 2.1 资产收益率、均值向量与协方差矩阵
- \(r_t,\mu,V\)
- 组合收益与组合方差
- 正定性与严格凸性语言

#### 2.2 Markowitz 均值—方差模型
- 最小方差问题
- 预算约束与收益约束
- 拉格朗日方法
- 有效前沿/双曲线背景

#### 2.3 无卖空约束与数值求解必要性
- \(x_i\ge 0\) 的现实含义
- 为什么解析解不再总是可得
- 与后续求解器的联系

#### 2.4 Sharpe 单指数模型
- 单指数收益分解
- 作为协方差结构简化背景
- 与本文主模型的关系：基础层而非主模型

#### 2.5 统一可行域与主记号
- long-only 可行域 \(\mathcal X\)
- 主变量 \(x\)、主矩阵 \(V\)、相关矩阵 \(C\)

### 核心公式
\[
r_{p,t}=x^\top r_t
\]
\[
\mu_p(x)=x^\top \mu
\]
\[
\sigma_p^2(x)=x^\top Vx
\]
\[
\mathcal X=\{x:\mathbf 1^\top x=1,\ 0\le x_i\le x_{\max}\}
\]

### 需要说明/可证明的命题
- 命题 2.1：在 \(V\succ 0\) 下，Markowitz 二次目标严格凸。
- 命题 2.2：加入无卖空约束后，最优解一般需要数值方法求得。

### 图表
- 图 2：ETF universe 与资产类别映射图
- 图 3：样本期代表性相关矩阵热图（原始版）
- 表 2：主要符号表（来自 `01_notation_master.md`）

### 实验
- 数据描述统计
- ETF 资产类别覆盖说明
- 样本区间与调仓频率说明

### 理论依据
本章直接对应笔记中对收益率、协方差矩阵、Markowitz、无卖空约束和 Sharpe 模型的整理。 :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

---

## 第 3 章 CtR 的表征理论与 long-only 可操作化

### 使命
完整建立 CtR 的理论对象、表征目标和求解思路，并在本章内部完成 long-only 下的可操作化。  
这一章是全文第一主线章。

### 本章必须回答的问题
1. 什么是 CtR？
2. ERC / risk budgeting 如何由 CtR 导出？
3. 为什么 CtR 的表征目标具有良好的凸性/强凸性结构？
4. 为什么在 long-only 下 CtR 可以直接保留并转成偏离度 \(D_R\)？

### 核心内容
#### 3.1 风险贡献的定义与 Euler 分解
- 边际风险贡献
- 总风险贡献
- 标准化 CtR
- \(\sum_i \mathrm{CtR}_i(x)=1\)

#### 3.2 Equal Risk Contribution 与风险预算
- 风险预算向量 \(b\)
- ERC 作为 \(b_i=1/n\) 的特例
- CtR 平价的经济含义

#### 3.3 CtR 的表征目标与一阶条件
- 笔记中的对数势函数 / 辅助函数
- 一阶最优性条件
- 归一化问题

#### 3.4 CtR 表征目标的凸性、强凸性与存在性
- 何时严格凸
- 何时强凸
- 最值存在性的逻辑
- 符号约束与定义域

#### 3.5 CtR 的数值求解思想：先梯度后牛顿
- 为什么先梯度
- 为什么后牛顿
- 为什么不能一开始直接牛顿

#### 3.6 CtR 在 long-only 下的保留与可操作化
- long-only 下 CtR 仍然定义良好
- 预算对象 \(b\) 仍然有意义
- 定义 CtR 偏离度
  \[
  D_R(x;b)=\sum_i(\mathrm{CtR}_i(x)-b_i)^2
  \]
- 说明 \(D_R\) 是 CtR 理论对象在 long-only 下的直接 operationalization

### 核心公式
\[
\mathrm{CtR}_i(x)=\frac{x_i(Vx)_i}{x^\top Vx}
\]
\[
\sum_{i=1}^n \mathrm{CtR}_i(x)=1
\]
\[
D_R(x;b)=\sum_{i=1}^n(\mathrm{CtR}_i(x)-b_i)^2
\]

### 需要证明/说明的命题
- 命题 3.1：CtR 构成组合总风险的加性分解。
- 命题 3.2：在适当定义域上，CtR 的表征目标具有严格凸/强凸结构。
- 命题 3.3：在 long-only 可行域上，CtR 作为预算型对象可直接保留，并自然诱导 \(D_R(x;b)\)。

### 图表
- 图 4：资本配置 vs 风险配置镜像条形图（EW / GMV / ERC）
- 图 5：CtR 时变热力图（ERC 与主模型）
- 图 6：CtR 收敛图（梯度阶段 + 牛顿阶段）

### 实验
- CtR-only / ERC 基准策略
- 与 EW、GMV 比较
- 输出：`ctr_long.csv`、`dr_db_timeseries.csv` 中的 `dr`

### 理论依据
本章直接基于笔记中的 CtR / ERC、强凸性、最值存在性与“先梯度后牛顿”思路展开。 :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

---

## 第 4 章 CtB 的表征理论与 long-only 可操作化

### 使命
完整建立 CtB 的理论对象、非线性系统、符号集合结构，并在本章内部完成 long-only 下的功能量转换。  
这一章是全文第二主线章。

### 本章必须回答的问题
1. 什么是 CtB？
2. 为什么 CtB 理论天然更依赖相关结构、符号集合与非线性系统？
3. 为什么某些 CtB 表征目标在 long-only simplex 上会退化？
4. 为什么 \(D_B\) 是 long-only 下最自然的 CtB 功能量？

### 核心内容
#### 4.1 CtB 的定义与相关暴露含义
- 资产与组合的相关暴露
- CtB 作为 correlation to basket
- CtB 与组合结构的关系

#### 4.2 CtB 的表征目标与相关性相等问题
- 一般权重下的 CtB 表征目标
- 相关性相等条件
- 与相关矩阵 / 协方差矩阵的联系

#### 4.3 变量替换与非线性系统
- 笔记中的变量替换
- 非线性系统的构造
- 为什么 CtB parity 不是简单二次优化

#### 4.4 符号集合、多极值与局部/全局结构
- 长头 / 短头集合
- 多极值
- 不同集合上问题的性质

#### 4.5 CtB 的数值求解思想：先梯度后牛顿
- 与 CtR 类似但更敏感
- 为什么需要先在可行区域内稳定下降

#### 4.6 CtB 表征目标在 long-only 下的退化
- 说明依赖 \(|x_i|\) 或符号结构的 surrogate 在 simplex 上退化
- 说明原表征目标不再适合作为经验主模型目标

#### 4.7 CtB 在 long-only 下的可操作化
- 回到 CtB 的原始对象
  \[
  \mathrm{CtB}_i(x)=\frac{(Vx)_i}{\sigma_i\sigma_p(x)}
  \]
- 定义 CtB 横截面离散度
  \[
  D_B(x)=\sum_i(\mathrm{CtB}_i(x)-\overline{\mathrm{CtB}}(x))^2
  \]
- 说明 \(D_B\) 是 CtB 理论在 long-only 下的正确 functional

### 核心公式
\[
\mathrm{CtB}_i(x)=\frac{(Vx)_i}{\sigma_i\sigma_p(x)}
\]
\[
\overline{\mathrm{CtB}}(x)=\frac1n\sum_i\mathrm{CtB}_i(x)
\]
\[
D_B(x)=\sum_i(\mathrm{CtB}_i(x)-\overline{\mathrm{CtB}}(x))^2
\]

### 需要证明/说明的命题
- 命题 4.1：CtB 可写成资产收益与组合收益之间的相关系数。
- 命题 4.2：CtB 表征问题在一般权重下对应非线性系统，并依赖符号集合。
- 命题 4.3：在 long-only simplex 上，部分 CtB surrogate 退化，因此必须回到 CtB 原始对象并定义 \(D_B\)。
- 命题 4.4：\(D_B\) 是 CtB 在 long-only 下的自然横截面离散度量。

### 图表
- 图 7：CtB 时变热力图（CtB-only 与主模型）
- 图 8：相关矩阵原图 + seriated / clustered 热图
- 图 9：CtB-only 求解收敛图

### 实验
- CtB-only 基准策略
- 与 ERC 比较
- 输出：`ctb_long.csv`、`dr_db_timeseries.csv` 中的 `db`

### 理论依据
本章直接对应笔记中的 correlation equality、CtB、变量替换、非线性系统、符号集合、多极值和梯度+牛顿方法。 :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

---

## 第 5 章 CtR–CtB 关系、不可兼容性与协调模型选择

### 使命
这一章只做三件事：
1. 写清 CtR–CtB 的关系；
2. 说明一般情形下为何不能同时严格平价；
3. 在此基础上选择最终主模型。

这一章必须短而硬，不再承担前面章节的补救解释。

### 本章必须回答的问题
1. CtR 与 CtB 在数学上到底是什么关系？
2. 为什么二者一般不能同时严格平价？
3. 为什么主模型应选“CtR 主目标 + CtB 约束”？

### 核心内容
#### 5.1 CtR–CtB 关系式
由定义直接推得：
\[
\mathrm{CtR}_i(x)=\frac{x_i\sigma_i}{\sigma_p(x)}\mathrm{CtB}_i(x)
\]
说明二者来自同一 \((Vx)_i\)，只是归一化不同。

#### 5.2 一般情形下不能同时严格平价
- 若同时要求 CtR 相等与 CtB 相等，则必须满足额外强条件
- 一致只发生在特殊结构下
- 一般情形下应接受“不能双严格平价”的事实

#### 5.3 协调模型的选择
- CtR 是预算型对象，适合作主目标
- CtB 是相关暴露型对象，适合作结构性约束
- 因而主模型取：
  \[
  \min_{x\in\mathcal X}
  D_R(x;b)+\eta\|x-x_{t-1}\|_2^2+\gamma\|x\|_2^2
  \quad
  \text{s.t.}\quad D_B(x)\le \delta
  \]
- 实现型：
  \[
  J(x)=D_R(x;b)+\eta\|x-x_{t-1}\|_2^2+\gamma\|x\|_2^2+\frac{\rho}{2}(D_B(x)-\delta)_+^2
  \]

### 核心公式
\[
\mathrm{CtR}_i(x)=\frac{x_i\sigma_i}{\sigma_p(x)}\mathrm{CtB}_i(x)
\]
\[
\min_{x\in\mathcal X}
D_R(x;b)+\eta\|x-x_{t-1}\|_2^2+\gamma\|x\|_2^2
\quad
\text{s.t.}\quad D_B(x)\le \delta
\]

### 需要证明/说明的命题
- 命题 5.1：CtR 与 CtB 由同一风险传播项驱动，但归一化不同。
- 命题 5.2：一般情形下 CtR parity 与 CtB parity 不能同时严格实现。
- 命题 5.3：在 long-only 世界中，最自然的协调模型是 CtR 主目标 + CtB 结构约束。

### 图表
- 图 10：\(D_R\)–\(D_B\) 双机制前沿图
- 图 11：CtR / CtB 横截面对照图（代表时点）

### 实验
- ERC、CtB-only、主模型三者的机制位置比较
- 不需要长篇回测，重点是机制图

### 理论依据
本章直接建立在笔记中 CtR 与 CtB 的关系和一致条件讨论之上。 :contentReference[oaicite:9]{index=9}

---

## 第 6 章 统一求解器与数值实现

### 使命
把笔记中的“先梯度、后牛顿”上升为全文统一求解器框架，并覆盖 CtR-only、CtB-only 与主模型。

### 本章必须回答的问题
1. 为什么统一求解器比三套独立 solver 更好？
2. 为什么先梯度后牛顿？
3. 如何在 long-only 可行域中实现可行投影与 active-set Newton？

### 核心内容
#### 6.1 统一问题表述
- 三类模型共享同一可行域与接口

#### 6.2 Projected Gradient 阶段
\[
x^{(k+1)}=\Pi_{\mathcal X}(x^{(k)}-\alpha_k\nabla J(x^{(k)}))
\]

#### 6.3 Damped Newton / Active-set 阶段
- 自由变量集
- 阻尼牛顿方向
- 可行线搜索

#### 6.4 收敛判据与失败回退
- KKT residual
- band violation
- fallback 机制

#### 6.5 代码实现结构
- solver API
- objective closure
- output diagnostics

### 核心公式
\[
x^{(k+1)}=\Pi_{\mathcal X}(x^{(k)}-\alpha_k g(x^{(k)}))
\]
\[
d_k=-(H(x^{(k)})+\mu_k I)^{-1}g(x^{(k)})
\]

### 需要证明/说明的命题
- 命题 6.1：在可行域约束下，先梯度后牛顿是比直接牛顿更稳健的两阶段框架。
- 命题 6.2：CtR-only、CtB-only 与主模型可由统一 solver family 覆盖。

### 图表
- 图 12：主模型收敛图
- 图 13：band violation / KKT residual 下降图
- 图 14：active-set 切换示意图（可选）

### 实验
- solver smoke tests
- run-time / convergence diagnostics
- 对比 SLSQP 或现有黑箱求解器作为基线（若需要）

### 理论依据
本章直接依据笔记中的“先梯度法、后牛顿法；直接牛顿可能越界”的思路。 :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

---

## 第 7 章 数据、参数校准与实验设计

### 使命
把“参数不是拍脑袋”写清楚，并固定 ETF 数据契约与实验协议。  
这一章是实证可信度的核心。

### 本章必须回答的问题
1. ETF universe 如何选？
2. 数据预处理与样本切分如何定？
3. \(\delta,\eta,\gamma,\rho\) 如何分层处理？
4. 实验如何保证可复现？

### 核心内容
#### 7.1 ETF universe 与样本期
- 大类资产 ETF 列表
- 数据频率与调仓频率
- 样本区间

#### 7.2 风险输入与协方差估计
- \(V_t\) 的估计方式
- rolling / expanding 设定
- shrinkage 设定（如采用）

#### 7.3 参数分层
- 固定结构参数：\(b=1/n,\ x_{\max}\)
- 校准参数：\(\delta,\eta\)
- 数值参数：\(\gamma,\rho\)

#### 7.4 校准协议
- train / validation / test
- \(\delta\) 的 quantile × scale family
- \(\eta\) 由 turnover target 反推
- \(\gamma\) 取最小稳定值
- \(\rho\) 取足够大且固定

#### 7.5 对照组与评价指标
- EW
- GMV
- ERC
- CtB-only
- 主模型

### 核心公式
\[
\delta \in \{\kappa \cdot Q_p(D_B^{ERC,\ train})\}
\]
\[
\eta \ \text{通过平均换手目标反推}
\]

### 需要说明的命题
- 命题 7.1：主模型的真正经济自由参数只有少数几个。
- 命题 7.2：参数协议是规则化选择，不是任意调参。

### 图表
- 图 15：ETF universe 分类图
- 图 16：校准可行域热力图
- 表 3：参数分层表
- 表 4：实验注册表

### 实验
- data audit
- calibration sweep
- stability checks

---

## 第 8 章 实证结果与机制分析

### 使命
展示主模型是否真的实现了“CtR 主配置 + CtB 结构控制”，并给出绩效与机制双重证据。  
这是全文的结果章。

### 本章必须回答的问题
1. 主模型是否改善了 CtB 结构而没有过度破坏 CtR 预算？
2. 主模型绩效是否优于或至少不劣于主要基准？
3. 机制改善是怎样发生的？
4. 是否存在明显的稳定性代价或换手代价？

### 核心内容
#### 8.1 总体绩效比较
- return / vol / Sharpe / drawdown
- turnover

#### 8.2 CtR 机制结果
- CtR 分布
- \(D_R\) rolling path
- CtR heatmap

#### 8.3 CtB 机制结果
- CtB 分布
- \(D_B\) rolling path
- CtB heatmap
- band active windows

#### 8.4 协调效果
- \(D_R\)–\(D_B\) frontier
- 主模型相对 ERC 的结构改善
- 约束活跃时段分析

#### 8.5 稳健性检验
- 不同窗口
- 不同 universe 子集
- 不同 \(x_{\max}\)
- 不同参数组

### 核心公式
本章以指标定义为主，不新增理论公式。

### 图表（主文核心图）
- 图 17：资本配置 vs 风险配置镜像条形图
- 图 18：累计净值与回撤图
- 图 19：CtR 热力图
- 图 20：CtB 热力图
- 图 21：rolling \(D_R/D_B\) 双面板 + active shading
- 图 22：\(D_R\)–\(D_B\) 双机制前沿图
- 图 23：相关矩阵热图 + seriated 热图
- 图 24：主模型约束活跃时段图

### 表格
- 表 5：总体绩效比较表
- 表 6：机制指标比较表
- 表 7：稳健性检验表

### 实验输出文件
- `summary_metrics.csv`
- `weights.csv`
- `ctr_long.csv`
- `ctb_long.csv`
- `dr_db_timeseries.csv`
- `objective_terms.csv`
- `diagnostics.json`
- `analysis_pack.json`

---

## 第 9 章 结论与展望

### 使命
压缩全文结论，并指出研究边界与后续扩展方向。  
结论必须与前面所有章节一一对应。

### 本章必须回答的问题
1. CtR 与 CtB 在本文中分别扮演了什么角色？
2. 为什么 long-only 下需要从理论表征走向可操作功能量？
3. 为什么最终主模型是 CtR 主目标 + CtB 约束？
4. 本文的局限与未来扩展是什么？

### 核心内容
#### 9.1 主要结论
- CtR 理论对象可直接 long-only 化
- CtB surrogate 退化，但 CtB 原始对象可保留并转成功能量
- 一般情形下不能双严格平价
- 主模型是最自然的协调方案

#### 9.2 研究贡献回顾
- 理论层
- 可操作化层
- 模型层
- 求解器层
- 实证层

#### 9.3 局限与展望
- 更一般 universe
- 更复杂 risk estimators
- 因子层 / 主成分层扩展
- 交易成本显式建模

### 图表
- 无新增核心图，可引用全文关键图回顾

### 实验
- 无新增实验

---

## 附录建议

### 附录 A
符号总表、定义总表（直接对应 `01_notation_master.md` 和 `02_definition_formula_ledger.md`）

### 附录 B
CtR 表征目标的凸性、强凸性、最值存在性详细证明

### 附录 C
CtB 非线性系统、变量替换、符号集合与多极值的详细推导

### 附录 D
统一求解器伪代码、梯度、Hessian、KKT 检查

### 附录 E
参数校准协议全文、实验注册表、附加稳健性结果