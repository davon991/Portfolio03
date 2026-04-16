# 02_definition_formula_ledger.md

## 用途说明
本台账用于统一：
- 论文名词
- 主定义公式
- 论文中的标准写法
- 代码变量名
- 结果文件字段名
- 出现章节
- 备注（是否属于理论表征、是否属于 long-only 可操作功能量）

---

## A. 基础对象

### A1. 资产收益率
- 论文名词：资产收益率
- 论文主记号：\(r_{i,t}\) / \(\xi_i\)（理论随机变量）
- 主定义：
  \[
  r_{i,t}=\frac{S_i(t)-S_i(t-1)}{S_i(t-1)}
  \]
  或理论写法 \(\xi_i\)
- 代码变量名：`returns`, `r_t`
- 结果字段名：`ret`
- 章节：第2章
- 备注：主线统一用 `r_{i,t}`；随机变量推导可局部用 \(\xi_i\)。 :contentReference[oaicite:16]{index=16} :contentReference[oaicite:17]{index=17}

### A2. 均值向量
- 论文名词：期望收益率向量
- 论文主记号：\(\mu\)
- 主定义：
  \[
  \mu=\mathbb E[r_t]
  \]
- 代码变量名：`mu`
- 结果字段名：无（通常不单独输出）
- 章节：第2章
- 备注：若使用单资产记号，则 \(a_i=\mathbb E[\xi_i]\)。 :contentReference[oaicite:18]{index=18}

### A3. 协方差矩阵
- 论文名词：协方差矩阵
- 论文主记号：\(V\)
- 主定义：
  \[
  V=(\sigma_{ij})_{n\times n},
  \qquad
  \sigma_{ij}=\mathrm{cov}(r_i,r_j)
  \]
- 代码变量名：`cov`
- 结果字段名：通常不直接输出，可写入 `cov_meta.json`
- 章节：第2章、第3章、第4章
- 备注：不再用 `A` 作为主记号，除非局部说明 \(A=V\)。 :contentReference[oaicite:19]{index=19} :contentReference[oaicite:20]{index=20}

### A4. 相关矩阵
- 论文名词：相关矩阵
- 论文主记号：\(C\)
- 主定义：
  \[
  C=(\rho_{ij})_{n\times n}
  \]
- 代码变量名：`corr`
- 结果字段名：通常不直接输出，可写入 `corr_meta.json`
- 章节：第4章
- 备注：笔记中 CtB 理论会涉及 correlation structure。 :contentReference[oaicite:21]{index=21}

### A5. 权重向量
- 论文名词：组合权重向量
- 论文主记号：\(x\)
- 主定义：
  \[
  x=(x_1,\dots,x_n)^\top
  \]
- 代码变量名：`x`, `weights`
- 结果字段名：`weight`
- 章节：全篇
- 备注：实证主线中 `x` 只表示权重，不表示持仓数量。 :contentReference[oaicite:22]{index=22}

### A6. 组合收益率
- 论文名词：组合收益率
- 论文主记号：\(r_{p,t}\)
- 主定义：
  \[
  r_{p,t}=x^\top r_t
  \]
- 代码变量名：`port_ret`
- 结果字段名：`portfolio_return`
- 章节：第2章
- 备注：理论中也可写成 \(\xi_p\)。 :contentReference[oaicite:23]{index=23} :contentReference[oaicite:24]{index=24}

### A7. 组合方差
- 论文名词：组合方差
- 论文主记号：\(\sigma_p^2(x)\)
- 主定义：
  \[
  \sigma_p^2(x)=x^\top Vx
  \]
- 代码变量名：`port_var`
- 结果字段名：`portfolio_variance`
- 章节：第2章、第3章、第4章
- 备注：组合波动率记为 \(\sigma_p(x)=\sqrt{x^\top Vx}\)。 :contentReference[oaicite:25]{index=25}

---

## B. CtR 体系

### B1. 边际风险贡献
- 论文名词：边际风险贡献
- 论文主记号：\(\mathrm{MRC}_i(x)\)
- 主定义：
  \[
  \mathrm{MRC}_i(x)=\frac{(Vx)_i}{\sigma_p(x)}
  \]
- 代码变量名：`mrc`
- 结果字段名：通常不单独输出
- 章节：第3章
- 备注：用于推导 CtR。 :contentReference[oaicite:26]{index=26}

### B2. 总风险贡献
- 论文名词：总风险贡献
- 论文主记号：\(\mathrm{RC}_i(x)\)
- 主定义：
  \[
  \mathrm{RC}_i(x)=x_i\mathrm{MRC}_i(x)=\frac{x_i(Vx)_i}{\sigma_p(x)}
  \]
- 代码变量名：`rc`
- 结果字段名：通常不单独输出
- 章节：第3章
- 备注：作为 CtR 的未标准化版本。 :contentReference[oaicite:27]{index=27}

### B3. CtR
- 论文名词：风险贡献占比 / Contribution to Risk
- 论文主记号：\(\mathrm{CtR}_i(x)\)
- 主定义：
  \[
  \mathrm{CtR}_i(x)=\frac{x_i(Vx)_i}{x^\top Vx}
  \]
- 代码变量名：`ctr`
- 结果字段名：`ctr`
- 章节：第3章、第5章、第8章
- 备注：主线统一采用标准化版本。 :contentReference[oaicite:28]{index=28}

### B4. 风险预算向量
- 论文名词：风险预算向量
- 论文主记号：\(b\)
- 主定义：
  \[
  b=(b_1,\dots,b_n)^\top,\quad b_i>0,\ \sum_i b_i=1
  \]
- 代码变量名：`budget`
- 结果字段名：通常不直接输出，可写入配置文件
- 章节：第3章、第6章
- 备注：正文主线默认 \(b_i=1/n\)。 

### B5. CtR 偏离度
- 论文名词：CtR 偏离度
- 论文主记号：\(D_R(x;b)\)
- 主定义：
  \[
  D_R(x;b)=\sum_i(\mathrm{CtR}_i(x)-b_i)^2
  \]
- 代码变量名：`dr`
- 结果字段名：`dr`
- 章节：第3章、第6章、第8章
- 备注：CtR 理论在 long-only 下的可操作功能量。([raw.githubusercontent.com](https://raw.githubusercontent.com/davon991/Portfolio/main/paper/spec.md))

### B6. ERC
- 论文名词：等风险贡献 / Equal Risk Contribution
- 论文主记号：ERC
- 主定义：
  \[
  b_i=\frac1n
  \]
  或等价地 \(\mathrm{CtR}_i(x)=1/n\)
- 代码变量名：`erc`
- 结果字段名：`strategy=ERC`
- 章节：第3章、第8章
- 备注：CtR-only 基准策略。 :contentReference[oaicite:29]{index=29}

---

## C. CtB 体系

### C1. CtB
- 论文名词：相关性贡献 / Correlation to Basket
- 论文主记号：\(\mathrm{CtB}_i(x)\)
- 主定义：
  \[
  \mathrm{CtB}_i(x)=\mathrm{Corr}(r_i,r_p)
  =\frac{(Vx)_i}{\sigma_i\,\sigma_p(x)}
  \]
- 代码变量名：`ctb`
- 结果字段名：`ctb`
- 章节：第4章、第5章、第8章
- 备注：主线采用 long-only 下的原始 CtB 对象。 :contentReference[oaicite:30]{index=30} :contentReference[oaicite:31]{index=31}

### C2. CtB 横截面均值
- 论文名词：CtB 横截面均值
- 论文主记号：\(\overline{\mathrm{CtB}}(x)\)
- 主定义：
  \[
  \overline{\mathrm{CtB}}(x)=\frac1n\sum_i\mathrm{CtB}_i(x)
  \]
- 代码变量名：`ctb_mean`
- 结果字段名：通常不单独输出
- 章节：第4章
- 备注：用于定义 CtB dispersion。

### C3. CtB 离散度
- 论文名词：CtB 离散度
- 论文主记号：\(D_B(x)\)
- 主定义：
  \[
  D_B(x)=\sum_i(\mathrm{CtB}_i(x)-\overline{\mathrm{CtB}}(x))^2
  \]
- 代码变量名：`db`
- 结果字段名：`db`
- 章节：第4章、第6章、第8章
- 备注：CtB 理论在 long-only 下的可操作功能量。([raw.githubusercontent.com](https://raw.githubusercontent.com/davon991/Portfolio/main/paper/spec.md))

### C4. CtB-only 基准
- 论文名词：CtB-only 基准
- 论文主记号：\(x_{B\min}\)
- 主定义：
  \[
  x_{B\min}\in\arg\min_{x\in\mathcal X} D_B(x)
  \]
- 代码变量名：`ctb_only`
- 结果字段名：`strategy=CTB_ONLY`
- 章节：第4章、第8章
- 备注：用于与 ERC 和主模型比较。

---

## D. CtR–CtB 关系

### D1. 桥接公式
- 论文名词：CtR–CtB 关系式
- 论文主记号：
  \[
  \mathrm{CtR}_i(x)=\frac{x_i\sigma_i}{\sigma_p(x)}\mathrm{CtB}_i(x)
  \]
- 代码变量名：通常不单独实现为字段，可在 analysis 模块中计算
- 结果字段名：无
- 章节：第5章
- 备注：用来说明二者一般不能同时严格平价。([raw.githubusercontent.com](https://raw.githubusercontent.com/davon991/Portfolio/main/paper/spec.md)) :contentReference[oaicite:32]{index=32}

---

## E. 主模型与参数

### E1. 可行域
- 论文名词：long-only 可行域
- 论文主记号：\(\mathcal X\)
- 主定义：
  \[
  \mathcal X=\{x:\mathbf 1^\top x=1,\ 0\le x_i\le x_{\max}\}
  \]
- 代码变量名：`feasible_set`（文档层）、实现中通常由约束表达
- 结果字段名：无
- 章节：第2章、第6章
- 备注：主实证固定采用此可行域。

### E2. CtB band 阈值
- 论文名词：CtB band 阈值
- 论文主记号：\(\delta\)
- 主定义：CtB dispersion 的上界
- 代码变量名：`delta_band`
- 结果字段名：`delta_band`（配置或 metadata）
- 章节：第6章、第7章
- 备注：由训练/验证协议校准。([raw.githubusercontent.com](https://raw.githubusercontent.com/davon991/Portfolio/main/paper/calibration_protocol.md))

### E3. 平滑参数
- 论文名词：平滑 / 换手控制参数
- 论文主记号：\(\eta\)
- 主定义：控制 \(\|x-x_{t-1}\|_2^2\) 项强度
- 代码变量名：`eta_smooth`
- 结果字段名：`eta_smooth`
- 章节：第6章、第7章

### E4. L2 稳定项
- 论文名词：数值稳定化参数
- 论文主记号：\(\gamma\)
- 主定义：控制 \(\|x\|_2^2\) 项强度
- 代码变量名：`gamma_l2`
- 结果字段名：`gamma_l2`
- 章节：第6章、第7章

### E5. hinge penalty 强度
- 论文名词：约束实现惩罚强度
- 论文主记号：\(\rho\)
- 主定义：控制 \((D_B-\delta)_+^2\) 项强度
- 代码变量名：`rho_penalty`
- 结果字段名：`rho_penalty`
- 章节：第6章、第7章

### E6. 主模型（概念型）
- 论文名词：协调模型 / 主模型
- 论文主记号：
  \[
  \min_{x\in\mathcal X} D_R(x;b)+\eta\|x-x_{t-1}\|_2^2+\gamma\|x\|_2^2
  \quad
  \text{s.t.}\quad D_B(x)\le \delta
  \]
- 代码变量名：`rb_ctb_band`
- 结果字段名：`strategy=RB_CTB_BAND`
- 章节：第5章、第6章、第8章
- 备注：主线固定为 CtR 主目标 + CtB 结构约束。([raw.githubusercontent.com](https://raw.githubusercontent.com/davon991/Portfolio/main/paper/spec.md))

### E7. 主模型（实现型目标）
- 论文名词：实现型目标函数
- 论文主记号：
  \[
  J(x)=D_R(x;b)+\eta\|x-x_{t-1}\|_2^2+\gamma\|x\|_2^2+\frac{\rho}{2}(D_B(x)-\delta)_+^2
  \]
- 代码变量名：`objective`
- 结果字段名：`obj_total`
- 章节：第6章
- 备注：论文算法实现与代码实现统一使用该形式。

---

## F. 求解器对象

### F1. 梯度
- 论文名词：梯度
- 论文主记号：\(g(x)=\nabla J(x)\)
- 代码变量名：`grad`
- 结果字段名：可选 `grad_norm`
- 章节：第6章
- 备注：Projected Gradient 阶段使用。

### F2. Hessian
- 论文名词：Hessian
- 论文主记号：\(H(x)=\nabla^2 J(x)\)
- 代码变量名：`hess`
- 结果字段名：不直接输出
- 章节：第6章

### F3. 迭代点
- 论文名词：第 \(k\) 次迭代权重
- 论文主记号：\(x^{(k)}\)
- 代码变量名：`xk`
- 结果字段名：可选 `iter_weights`
- 章节：第6章

### F4. 可行投影
- 论文名词：投影算子
- 论文主记号：\(\Pi_{\mathcal X}\)
- 代码变量名：`project_to_feasible`
- 结果字段名：无
- 章节：第6章

### F5. 收敛诊断
- 论文名词：收敛诊断
- 论文主记号：无统一数学主记号
- 代码变量名：`diagnostics`
- 结果字段名：
  - `converged`
  - `iterations`
  - `kkt_residual`
  - `band_active`
- 章节：第6章、第8章

---

## G. Sharpe 模型对象（基础层）

### G1. 指数收益率
- 论文名词：市场指数收益率
- 论文主记号：\(I_t\)
- 代码变量名：`index_ret`
- 结果字段名：无
- 章节：第2章或附录
- 备注：仅作基础模型扩展。 :contentReference[oaicite:33]{index=33}

### G2. 单指数模型
- 论文名词：Sharpe 单指数模型
- 论文主记号：
  \[
  r_{i,t}=\alpha_i+\beta_i I_t+\varepsilon_{i,t}
  \]
- 代码变量名：`alpha`, `beta`, `eps_var`
- 结果字段名：可选 `alpha_hat`, `beta_hat`
- 章节：第2章或附录
- 备注：不进入主模型主线。 :contentReference[oaicite:34]{index=34}

---

## H. 结果文件字段总映射（建议固定）

### H1. strategy-level summary
- `strategy`
- `ann_return`
- `ann_vol`
- `sharpe`
- `max_drawdown`
- `turnover_mean`
- `turnover_p95`
- `dr_mean`
- `db_mean`
- `active_rate`

### H2. weights.csv
- `date`
- `strategy`
- `asset`
- `weight`

### H3. ctr_long.csv
- `date`
- `strategy`
- `asset`
- `ctr`

### H4. ctb_long.csv
- `date`
- `strategy`
- `asset`
- `ctb`

### H5. dr_db_timeseries.csv
- `date`
- `strategy`
- `dr`
- `db`
- `band_active`

### H6. objective_terms.csv
- `date`
- `strategy`
- `obj_total`
- `dr_term`
- `smooth_term`
- `l2_term`
- `band_penalty`

### H7. diagnostics.json
- `converged`
- `iterations`
- `kkt_residual_summary`
- `violation_counts`
- `active_constraints`

---

## I. 冻结规则

1. 同一对象只能保留一个主记号。  
2. 若后续模块需要引入辅助记号，必须说明其与主记号的映射。  
3. 若论文、代码、结果字段三者出现不一致，优先修正代码变量名与结果字段名以服从本台账。  
4. 未经明确讨论，不得新增与主线无关的对象名。