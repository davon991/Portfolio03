# 06_model_contract.md

## 1. 文件使命

本文件冻结本项目的模型层。  
从本文件起，后续论文、代码、图表、实验注册、参数校准和答辩材料，必须使用本文件中给定的：

- 理论对象；
- long-only 下的可操作功能量；
- 基准模型；
- 主模型；
- 参数分层；
- 模型输入输出契约。

本文件的核心目标是避免以下漂移：

1. 把 CtR、CtB 理论对象与主模型对象混淆；
2. 把 long-only 下退化的 CtB surrogate 继续误当主模型目标；
3. 在实验阶段临时新增无解释的参数；
4. 在代码与论文中使用不同模型口径。

---

## 2. 模型层总原则

### 2.1 两层对象体系
本项目的模型层统一采用“两层对象体系”：

#### 第一层：理论对象（representation objects）
这一层直接承接用户笔记中的 CtR / CtB 理论定义与表征目标。  
作用是：
- 定义“什么是 CtR”
- 定义“什么是 CtB”
- 给出 CtR / CtB 平价的理论条件
- 给出最优性、凸性、符号集合、非线性系统等理论结构
- 支撑“先梯度后牛顿”的数值思想

#### 第二层：long-only 可操作功能量（operational functionals）
这一层用于真正的 long-only 实证模型。  
作用是：
- 在 long-only 可行域上定义可计算、可比较、可约束的目标量；
- 把理论对象转化为基准策略和主模型的优化功能量。

### 2.2 三层模型体系
本项目固定存在三类模型：

1. **CtR-only 基准**
2. **CtB-only 基准**
3. **协调主模型**

它们不是三套互相无关的系统，而是同一理论框架下的三种角色：

- CtR-only：预算平衡端点
- CtB-only：相关结构平衡端点
- 主模型：在 long-only 世界中对二者进行协调

### 2.3 实证世界固定为 long-only
所有正式实验模型一律在以下可行域中求解：

\[
\mathcal X=\{x\in\mathbb R^n:\mathbf 1^\top x=1,\ 0\le x_i\le x_{\max}\}.
\]

其中：
- \(\mathbf 1^\top x=1\)：全投资
- \(x_i\ge 0\)：long-only
- \(x_i\le x_{\max}\)：单资产权重上限

---

## 3. CtR 理论对象（冻结）

### 3.1 CtR 的主定义
定义标准化风险贡献：

\[
\mathrm{CtR}_i(x)=\frac{x_i(Vx)_i}{x^\top Vx},
\qquad i=1,\dots,n.
\]

满足：
\[
\sum_{i=1}^n \mathrm{CtR}_i(x)=1.
\]

### 3.2 风险预算
定义风险预算向量：

\[
b=(b_1,\dots,b_n)^\top,\qquad b_i>0,\ \sum_{i=1}^n b_i=1.
\]

正文主线默认采用等预算：
\[
b_i=\frac1n,\qquad i=1,\dots,n.
\]

### 3.3 CtR 理论角色
CtR 是本项目的**预算型理论对象**，承担以下职能：

1. 刻画组合总风险如何在资产之间分配；
2. 连接 ERC / risk budgeting 主线；
3. 提供 long-only 下最自然的主配置目标。

### 3.4 CtR 理论表征目标
CtR 理论章节允许使用笔记中的对数势函数 / 辅助函数进行推导，例如：

\[
\Phi_R(y)=\frac12 y^\top V y-\tau \sum_{i=1}^n b_i \ln y_i,
\qquad y_i>0.
\]

该对象的角色是：
- 用于推导 CtR / risk budgeting；
- 用于讨论凸性、强凸性、存在性；
- 用于引出“先梯度后牛顿”的求解思想。

**冻结规定：**  
\(\Phi_R\) 属于理论表征目标，不直接作为主实证模型目标函数；  
它在模型层的作用是“生成 CtR 对象与理论结构”。

---

## 4. CtB 理论对象（冻结）

### 4.1 CtB 的主定义
定义第 \(i\) 个资产与组合收益率之间的相关暴露：

\[
\mathrm{CtB}_i(x)=\mathrm{Corr}(r_i,r_p)
=\frac{(Vx)_i}{\sigma_i\,\sigma_p(x)},
\qquad i=1,\dots,n.
\]

其中：
\[
\sigma_i=\sqrt{V_{ii}},\qquad
\sigma_p(x)=\sqrt{x^\top Vx}.
\]

### 4.2 CtB 理论角色
CtB 是本项目的**相关暴露型理论对象**，承担以下职能：

1. 刻画每个资产与组合总波动方向的相关暴露；
2. 对应相关结构 / diversification / equal-correlation 思想；
3. 提供主模型中的结构性控制对象。

### 4.3 CtB 理论表征目标
CtB 理论章节允许保留笔记中的一般权重/符号集合下的 surrogate 或表征目标，例如：

\[
\Phi_B(x)=\frac12 x^\top C x-\sum_{i=1}^n |x_i|
\]

或笔记中的等价规范化表述。  
该对象的角色是：

- 用于生成 CtB parity 的理论条件；
- 用于说明 CtB 问题的非线性系统；
- 用于讨论符号集合、多极值和局部/全局结构；
- 用于支撑 CtB 章节中的“先梯度后牛顿”求解思想。

### 4.4 long-only 下的关键结论
在 long-only 且全投资的 simplex 上：
\[
x_i\ge 0,\qquad \mathbf 1^\top x=1
\]
时，若 surrogate 中含有 \(\sum_i |x_i|\) 这样的项，则有：
\[
\sum_i |x_i|=\sum_i x_i=1,
\]
从而该部分退化为常数。

**冻结结论：**  
CtB 的某些 signed surrogate / 表征目标在 long-only simplex 上会退化，因此它们不能直接作为主模型目标。

这一步不是“换题”，而是从理论表征退回到 CtB 的原始对象，再进行 long-only 可操作化。

---

## 5. long-only 下的可操作功能量（冻结）

### 5.1 CtR 功能量
定义 CtR 偏离度：

\[
D_R(x;b)=\sum_{i=1}^n \bigl(\mathrm{CtR}_i(x)-b_i\bigr)^2.
\]

其角色是：

- CtR-only 基准模型的主目标；
- 主模型中的主配置目标；
- CtR 理论对象在 long-only 下的直接 operationalization。

### 5.2 CtB 功能量
定义 CtB 横截面均值：

\[
\overline{\mathrm{CtB}}(x)=\frac1n\sum_{i=1}^n \mathrm{CtB}_i(x).
\]

定义 CtB 离散度：

\[
D_B(x)=\sum_{i=1}^n \bigl(\mathrm{CtB}_i(x)-\overline{\mathrm{CtB}}(x)\bigr)^2.
\]

其角色是：

- CtB-only 基准模型的主目标；
- 主模型中的结构性约束对象；
- CtB 理论对象在 long-only 下的正确 functional。

### 5.3 冻结结论
本项目中：

- **CtR 理论对象 \(\to D_R\)**
- **CtB 理论对象 \(\to D_B\)**

是从理论层进入实证层的唯一合法路径。  
后续不得再引入其他未注册的“替代性 CtR/CtB 指标”。

---

## 6. CtR-only 基准模型（冻结）

### 6.1 模型定义
在 long-only 可行域 \(\mathcal X\) 上，定义 CtR-only 基准为：

\[
x_{CtR}(t)\in
\arg\min_{x\in\mathcal X}
D_R(x;b).
\]

正文主线默认采用 ERC 风险预算：
\[
b_i=\frac1n,\quad i=1,\dots,n.
\]

### 6.2 模型角色
CtR-only 基准的角色是：

1. 提供 risk budgeting / ERC 的 long-only 端点；
2. 作为主模型的比较基准；
3. 为参数校准中的 \(\delta\) 提供参考分布（例如 \(D_B\) 的 ERC baseline 分布）。

### 6.3 冻结命名
- 论文名：CtR-only 基准 / ERC 基准
- 代码模型名：`erc`
- 结果字段：`strategy = ERC`

---

## 7. CtB-only 基准模型（冻结）

### 7.1 模型定义
在 long-only 可行域 \(\mathcal X\) 上，定义 CtB-only 基准为：

\[
x_{CtB}(t)\in
\arg\min_{x\in\mathcal X}
D_B(x).
\]

### 7.2 模型角色
CtB-only 基准的角色是：

1. 提供相关暴露均衡的 long-only 端点；
2. 用于比较 CtB-only 和 CtR-only 在结构与风险分配上的差异；
3. 为主模型的 CtB 约束提供可达性下界和解释性对照。

### 7.3 冻结命名
- 论文名：CtB-only 基准
- 代码模型名：`ctb_only`
- 结果字段：`strategy = CTB_ONLY`

---

## 8. 主模型（冻结）

### 8.1 概念型主模型
本项目的正式主模型固定为：

\[
\min_{x\in\mathcal X}
D_R(x;b)
+\eta \|x-x_{t-1}\|_2^2
+\gamma \|x\|_2^2
\quad
\text{s.t.}\quad
D_B(x)\le \delta.
\]

### 8.2 实现型主模型
为了统一数值求解与结果导出，代码实现中采用 hinge penalty 版本：

\[
J(x)=
D_R(x;b)
+\eta \|x-x_{t-1}\|_2^2
+\gamma \|x\|_2^2
+\frac{\rho}{2}\bigl(D_B(x)-\delta\bigr)_+^2.
\]

其中：
\[
(z)_+ = \max(z,0).
\]

### 8.3 模型角色
主模型的角色是：

1. 以 CtR 偏离度 \(D_R\) 维持风险预算主线；
2. 以 CtB 离散度 \(D_B\) 控制相关暴露结构；
3. 以 \(\eta\) 控制换手和平滑性；
4. 以 \(\gamma\) 提供数值稳定与正则化；
5. 以 \(\rho\) 实现 band 约束的软化求解。

### 8.4 冻结命名
- 论文名：协调主模型 / CtR–CtB 协调模型
- 代码模型名：`rb_ctb_band`
- 结果字段：`strategy = RB_CTB_BAND`

---

## 9. 为什么主模型固定为 “CtR 主目标 + CtB 约束”

### 9.1 第一条理由：CtR 是预算对象
CtR 具有可加性：
\[
\sum_i \mathrm{CtR}_i(x)=1,
\]
并直接对应风险预算向量 \(b\)。  
因此 CtR 最适合作为组合配置的**主预算目标**。

### 9.2 第二条理由：CtB 是相关暴露对象
CtB 衡量的是每个资产与组合总波动方向的相关暴露。  
它反映的是“组合结构是否过度集中于某种共同波动方向”，因此更适合作为**结构性约束/护栏**，而不是主预算目标。

### 9.3 第三条理由：一般情形下二者不能同时严格平价
由关系式：
\[
\mathrm{CtR}_i(x)=\frac{x_i\sigma_i}{\sigma_p(x)}\,\mathrm{CtB}_i(x)
\]
可见，若要求所有活跃资产同时 CtR 相等且 CtB 相等，则必须满足额外强条件。  
因此一般情形下不应追求“双严格平价”，而应采用协调方案。

### 9.4 第四条理由：约束式比双目标加权更可解释
相对于
\[
\lambda D_R(x;b)+(1-\lambda)D_B(x)
\]
这种双目标加权写法，band 约束更容易解释为：

- 允许 CtB 在合理容忍带内波动；
- 但不允许其结构性失衡超过阈值 \(\delta\)。

因此：
- \(D_R\) 负责主配置；
- \(D_B\) 负责结构约束；
- 这比双目标混合更符合风险预算论文的主线与答辩逻辑。

### 9.5 冻结结论
主模型的正式建模选择，不是临时决策，而是模型层固定原则。  
后续论文必须在正文中明确解释这一选择，不允许只在答辩时口头补充。

---

## 10. 参数分层（冻结）

本项目所有模型参数严格分成三类：

### 10.1 固定参数（structural parameters）
这些参数属于结构设定，不参与主调参：

1. **风险预算向量**
   - 主线固定：
     \[
     b_i=\frac1n,\quad i=1,\dots,n.
     \]
   - 解释：正文主线采用 ERC 风险预算。

2. **可行域形式**
   - full investment
   - long-only
   - box constraint \(x_i\le x_{\max}\)

3. **单资产权重上限**
   - 记为 \(x_{\max}\)
   - 视作投资政策参数，不作为核心调参对象

4. **ETF universe 与调仓频率**
   - 由 `05_data_contract.md` 冻结

### 10.2 校准参数（economic / policy parameters）
这些参数需要规则化校准：

1. **CtB band 阈值**
   - \(\delta\)
   - 解释：允许的 CtB dispersion 上限
   - 校准原则：由 train / validation 规则化选择，不允许拍脑袋设定

2. **平滑/换手参数**
   - \(\eta\)
   - 解释：控制权重路径平滑度与换手
   - 校准原则：由目标平均换手或验证集性能反推

### 10.3 数值参数（numerical parameters）
这些参数不是策略主信念，而是实现参数：

1. **L2 稳定化参数**
   - \(\gamma\)
   - 作用：增强数值稳定与可解性
   - 原则：取最小稳定值

2. **hinge penalty 强度**
   - \(\rho\)
   - 作用：实现 band 约束
   - 原则：固定为足够大，不作为主经济参数解释

### 10.4 冻结结论
正文主线中真正需要解释的参数只有：

- \(\delta\)
- \(\eta\)

\(\gamma,\rho\) 必须在论文中明确降级为“数值参数”，不得写成主经济参数。  
这一步是为了防止主模型显得“参数太多、过于拍脑袋”。

---

## 11. 模型输入与输出契约

### 11.1 每个调仓时点的模型输入
模型层统一接收以下输入：

- `date_t`
- `asset_list`
- `x_prev`：上一期漂移后权重
- `cov_t = V_t`
- `corr_t = C_t`
- `sigma_t`
- `budget = b`
- `delta_band = \delta`
- `eta_smooth = \eta`
- `gamma_l2 = \gamma`
- `rho_penalty = \rho`
- `x_max`

### 11.2 每个调仓时点的模型输出
模型层统一输出：

- `x_opt`
- `ctr`
- `ctb`
- `dr`
- `db`
- `obj_total`
- `dr_term`
- `smooth_term`
- `l2_term`
- `band_penalty`
- `band_active`
- `solver_diagnostics`

### 11.3 与结果文件的映射
这些输出必须进一步写入：

- `weights.csv`
- `ctr_long.csv`
- `ctb_long.csv`
- `dr_db_timeseries.csv`
- `objective_terms.csv`
- `diagnostics.json`

输出字段名必须与 `02_definition_formula_ledger.md` 中登记的一致。

---

## 12. 与后续章节和代码模块的衔接

### 12.1 对第 3 章的要求
- 只允许把 CtR 理论对象导向 \(D_R\)
- 不允许在第 3 章中提前写主模型参数校准细节

### 12.2 对第 4 章的要求
- 必须明确说明 CtB surrogate 在 long-only 下的退化
- 只允许把 CtB 原始对象导向 \(D_B\)

### 12.3 对第 5 章的要求
- 不重复第 3、4 章理论
- 只做 CtR–CtB 关系、不能双严格平价、以及主模型选择

### 12.4 对代码模块的要求
代码中不允许存在：
- 未注册的第四类主模型；
- 未注册的 CtR / CtB 替代定义；
- 未说明来源的额外惩罚项；
- 把 \(\gamma,\rho\) 当成主调参参数来扫网格。

---

## 13. 冻结结论

从本文件起，本项目的模型层固定如下：

### 理论对象
- CtR
- CtB

### long-only 可操作功能量
- \(D_R(x;b)\)
- \(D_B(x)\)

### 三层模型
1. CtR-only：
   \[
   \min_{x\in\mathcal X} D_R(x;b)
   \]
2. CtB-only：
   \[
   \min_{x\in\mathcal X} D_B(x)
   \]
3. 主模型：
   \[
   \min_{x\in\mathcal X}
   D_R(x;b)+\eta\|x-x_{t-1}\|_2^2+\gamma\|x\|_2^2
   \quad
   \text{s.t.}\quad D_B(x)\le \delta
   \]

### 参数分层
- 固定参数：\(b,\mathcal X,x_{\max}\)
- 校准参数：\(\delta,\eta\)
- 数值参数：\(\gamma,\rho\)

### 主模型选择原则
- CtR 作为主目标
- CtB 作为结构约束
- 不采用无解释的双目标加权作为正文主模型

后续所有模块必须服从本契约。