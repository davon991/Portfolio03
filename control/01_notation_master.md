# 01_notation_master.md

## 1. 统一记号总原则

本项目统一采用以下记号体系：

1. **主变量统一用 `x` 表示投资组合权重向量**。  
   不再在主线中混用 `w`、`x`、`ω`、`N` 作为同层对象。
2. **收益率空间作为主工作空间**。  
   即主模型、主实证、CtR/CtB 定义、协方差矩阵、组合波动等，一律以收益率及其协方差矩阵为核心对象。
3. **以协方差矩阵 `V` 作为主风险矩阵记号**。  
   不在主线中混用 `A`、`U`、`Σ` 作为不同主记号；其中：
   - `V` 专指资产收益率协方差矩阵；
   - 若需要相关矩阵，统一记为 `C`；
   - 若在某段理论推导中沿用笔记中的二次型矩阵 `A`，正文应显式说明 `A=V` 或 `A=C`。
4. **CtR 与 CtB 是两个核心对象，不允许换名。**
5. **long-only 主模型的可行域固定为**
   \[
   \mathcal X=\{x\in\mathbb R^n:\mathbf 1^\top x=1,\ 0\le x_i\le x_{\max}\}.
   \]

---

## 2. 基础索引与维度

- \(n\)：资产个数
- \(i,j\)：资产索引，\(i,j=1,\dots,n\)
- \(t\)：时间索引
- \(\mathbf 1\)：全 1 向量，维度为 \(n\times 1\)

---

## 3. 资产与收益率记号

### 3.1 单资产价格与收益率
- \(S_i(t)\)：第 \(i\) 个资产在时点 \(t\) 的价格
- \(r_{i,t}\)：第 \(i\) 个资产在 \(t\) 期的收益率
- \(\xi_i\)：第 \(i\) 个资产收益率的随机变量表示（理论部分允许使用）
- \(a_i=\mathbb E[\xi_i]\)：第 \(i\) 个资产收益率的期望

### 3.2 向量形式
- \(r_t=(r_{1,t},\dots,r_{n,t})^\top\)：时点 \(t\) 的资产收益率向量
- \(\mu=(a_1,\dots,a_n)^\top\)：资产收益率均值向量

> 说明：  
> 笔记中曾使用 \(\xi_i\) 表示单资产收益率随机变量，也使用过 \(\Delta S_i\)、\(\Delta B\) 表示价格增量与组合增量。  
> 论文主线统一在“收益率—协方差矩阵—权重”框架下表达；  
> 若讨论 CtB 的原始相关暴露定义，可局部引入 \(\Delta S_i,\Delta B\) 作为解释性记号。 :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

---

## 4. 权重与组合记号

### 4.1 主记号
- \(x=(x_1,\dots,x_n)^\top\)：组合权重向量（主记号）
- \(x_i\)：第 \(i\) 个资产在组合中的权重

### 4.2 组合收益与组合波动
- \(r_{p,t}=x^\top r_t\)：组合在时点 \(t\) 的收益率
- \(\mu_p(x)=x^\top \mu\)：组合期望收益率
- \(\sigma_p^2(x)=x^\top Vx\)：组合方差
- \(\sigma_p(x)=\sqrt{x^\top Vx}\)：组合波动率

### 4.3 可行域
- long-only 可行域：
  \[
  \mathcal X=\{x\in\mathbb R^n:\mathbf 1^\top x=1,\ 0\le x_i\le x_{\max}\}
  \]

> 说明：  
> 笔记中 `N_i` 表示资产数量、`x_i` 或 `w_i` 也可表示资产占比。  
> 为避免论文、代码、结果文件混乱，本项目一律固定为：  
> **实证主线中 `x_i` 只表示权重，不表示持仓数量。** :contentReference[oaicite:5]{index=5} :contentReference[oaicite:6]{index=6}

---

## 5. 风险矩阵与相关矩阵

- \(V=(\sigma_{ij})_{n\times n}\)：资产收益率协方差矩阵
- \(\sigma_i^2=V_{ii}\)：第 \(i\) 个资产收益率方差
- \(\sigma_i=\sqrt{V_{ii}}\)：第 \(i\) 个资产收益率波动率
- \(C=(\rho_{ij})_{n\times n}\)：资产收益率相关矩阵
- \(\rho_{ij}\)：第 \(i,j\) 个资产收益率之间的相关系数

> 说明：  
> 笔记中既有协方差矩阵，也出现了相关矩阵和一般二次型矩阵；  
> 主线统一为：  
> - **`V` = covariance matrix**  
> - **`C` = correlation matrix**  
> 理论局部若出现 `A`，必须说明它具体代表 `V` 还是 `C`。 :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

---

## 6. CtR（Contribution to Risk）

### 6.1 主定义
定义边际风险贡献：
\[
\mathrm{MRC}_i(x)=\frac{\partial \sigma_p(x)}{\partial x_i}
=\frac{(Vx)_i}{\sigma_p(x)}.
\]

定义总风险贡献：
\[
\mathrm{RC}_i(x)=x_i\,\mathrm{MRC}_i(x)
=\frac{x_i(Vx)_i}{\sigma_p(x)}.
\]

定义标准化风险贡献（主线使用）：
\[
\mathrm{CtR}_i(x)=\frac{x_i(Vx)_i}{x^\top Vx}.
\]

于是：
\[
\sum_{i=1}^n \mathrm{CtR}_i(x)=1.
\]

### 6.2 风险预算
- \(b=(b_1,\dots,b_n)^\top\)：风险预算向量
- \(b_i>0,\ \sum_i b_i=1\)

### 6.3 ERC 特例
- 等风险预算：
  \[
  b_i=\frac1n,\quad i=1,\dots,n.
  \]

> 说明：  
> 笔记中 CtR 是从组合方差或组合波动的导数出发定义的，并进一步进入 Equal Risk Contribution。  
> 本项目主线统一采用**标准化贡献**
> \[
> \mathrm{CtR}_i(x)=\frac{x_i(Vx)_i}{x^\top Vx}
> \]
> 作为论文、代码和结果文件的主对象。 :contentReference[oaicite:9]{index=9}

---

## 7. CtB（Correlation to Basket）

### 7.1 原始定义
定义第 \(i\) 个资产与组合收益率之间的相关暴露：
\[
\mathrm{CtB}_i(x)=\mathrm{Corr}(r_i,r_p)
=\frac{(Vx)_i}{\sigma_i\,\sigma_p(x)}.
\]

### 7.2 横截面均值
\[
\overline{\mathrm{CtB}}(x)=\frac1n\sum_{i=1}^n \mathrm{CtB}_i(x).
\]

> 说明：  
> 笔记中 CtB 是通过“资产与组合（basket）之间的相关系数”定义的，并讨论了在一般权重、含多空/符号集合时的表征目标和非线性系统。  
> 论文主线统一把 CtB 的原始对象固定为上式。 :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

---

## 8. long-only 下的可操作功能量

### 8.1 CtR 偏离度
\[
D_R(x;b)=\sum_{i=1}^n\bigl(\mathrm{CtR}_i(x)-b_i\bigr)^2.
\]

其中 ERC 特例为：
\[
D_R(x;1/n)=\sum_{i=1}^n\left(\mathrm{CtR}_i(x)-\frac1n\right)^2.
\]

### 8.2 CtB 离散度
\[
D_B(x)=\sum_{i=1}^n\bigl(\mathrm{CtB}_i(x)-\overline{\mathrm{CtB}}(x)\bigr)^2.
\]

> 说明：  
> CtR 理论对象在 long-only 下可直接保留，因此 \(D_R\) 是 CtR 的直接 operationalization。  
> CtB 的某些 signed surrogate 在 simplex 上退化，因此主线不直接沿用其表征目标，而改用 \(D_B\) 作为 long-only 下的可操作功能量。  
> 这与项目总控中的主线一致。 :contentReference[oaicite:12]{index=12}

---

## 9. 协调模型参数

- \(\delta\)：CtB band 阈值
- \(\eta\)：平滑 / 换手控制参数
- \(\gamma\)：L2 数值稳定化参数
- \(\rho\)：hinge penalty 强度参数

### 9.1 主模型（概念型）
\[
\min_{x\in\mathcal X} D_R(x;b)+\eta\|x-x_{t-1}\|_2^2+\gamma\|x\|_2^2
\quad
\text{s.t.}\quad D_B(x)\le \delta.
\]

### 9.2 主模型（实现型）
\[
J(x)=D_R(x;b)+\eta\|x-x_{t-1}\|_2^2+\gamma\|x\|_2^2
+\frac{\rho}{2}\bigl(D_B(x)-\delta\bigr)_+^2.
\]

---

## 10. 优化与算法记号

- \(g(x)=\nabla J(x)\)：梯度
- \(H(x)=\nabla^2 J(x)\)：Hessian
- \(x^{(k)}\)：第 \(k\) 次迭代的权重向量
- \(\alpha_k\)：梯度步长
- \(d_k\)：牛顿方向
- \(\Pi_{\mathcal X}(\cdot)\)：投影到可行域 \(\mathcal X\)

### 10.1 梯度阶段
\[
x^{(k+1)}=\Pi_{\mathcal X}\bigl(x^{(k)}-\alpha_k g(x^{(k)})\bigr).
\]

### 10.2 牛顿阶段
\[
d_k=-\bigl(H(x^{(k)})+\mu_k I\bigr)^{-1}g(x^{(k)}),
\]
然后做可行步长搜索。

> 说明：  
> 这与笔记中“先做若干步梯度法，再切换到牛顿法；直接牛顿可能越界”的思想一致。 :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}

---

## 11. Sharpe 模型记号（仅作基础扩展，不进入主模型主线）

- \(I_t\)：市场指数收益率
- \(\alpha_i\)：第 \(i\) 个资产的特异项均值/常数项
- \(\beta_i\)：第 \(i\) 个资产对指数的暴露
- \(\varepsilon_{i,t}\)：白噪声项
- \(\sigma_{\varepsilon_i}^2\)：特异风险方差

单指数表示：
\[
r_{i,t}=\alpha_i+\beta_i I_t+\varepsilon_{i,t}.
\]

> 说明：  
> Sharpe 模型来自笔记，保留为第 2 章或附录中的基础扩展，不直接作为主模型核心对象。  
> 相关记号应与主线中的 \(x,\mu,V\) 保持兼容。 :contentReference[oaicite:15]{index=15}

---

## 12. 代码变量命名总原则

### 12.1 主变量
- `x`：权重向量
- `mu`：均值向量
- `cov`：协方差矩阵 \(V\)
- `corr`：相关矩阵 \(C\)

### 12.2 核心对象
- `ctr`：CtR 向量
- `ctb`：CtB 向量
- `dr`：CtR 偏离度 \(D_R\)
- `db`：CtB 离散度 \(D_B\)

### 12.3 参数
- `budget`：风险预算向量 \(b\)
- `delta_band`：\(\delta\)
- `eta_smooth`：\(\eta\)
- `gamma_l2`：\(\gamma\)
- `rho_penalty`：\(\rho\)

### 12.4 结果字段
- `weight`
- `ctr`
- `ctb`
- `dr`
- `db`
- `band_active`
- `obj_total`
- `turnover`

---

## 13. 冻结结论

从本文件起，本项目统一采用：

- 主权重记号：`x`
- 主风险矩阵：`V`
- 主相关矩阵：`C`
- 主风险贡献对象：`CtR`
- 主相关暴露对象：`CtB`
- CtR 功能量：`D_R`
- CtB 功能量：`D_B`

后续论文、代码、图表、表格、JSON / CSV 输出，均必须遵守本文件。