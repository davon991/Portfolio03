# 07_solver_contract.md

## 1. 文件使命

本文件冻结本项目的统一求解器设计。
从本文件起，后续代码实现、实验运行、结果导出、论文写作与答辩说明，必须以本文件为唯一 solver 契约来源。

本文件的任务不是直接给出最终代码，而是一次性固定以下内容：

1. 求解器的统一接口；
2. CtR-only、CtB-only、主模型三类问题如何在同一 solver family 中实现；
3. “先梯度、后牛顿”的两阶段算法如何体现；
4. 可行域维护、active-set 处理、步长规则、收敛判据与失败回退；
5. 代码模块和结果文件必须输出哪些诊断量。

本文件严格服从：

* `00_project_charter.md`
* `01_notation_master.md`
* `02_definition_formula_ledger.md`
* `05_data_contract.md`
* `06_model_contract.md`

---

## 2. 设计原则（冻结）

### 2.1 单一 solver family 原则

本项目**不允许**为 CtR-only、CtB-only 和主模型分别设计三套互不相干的求解器。
统一采用一个 solver family：

* 统一可行域；
* 统一接口；
* 统一两阶段流程；
* 不同模型只通过 objective closure / gradient closure / Hessian closure 区分。

### 2.2 两阶段原则

统一求解器必须体现用户笔记中的核心思想：

1. **先梯度法（Projected Gradient）**：

   * 作用：在可行域内稳健下降；
   * 目的：避免一开始直接牛顿导致越界、穿越边界或落入不稳定区域。

2. **后牛顿法（Active-set Damped Newton）**：

   * 作用：在接近最优点时加速收敛；
   * 目的：提高精度和局部收敛速度。

### 2.3 long-only 可行域原则

所有正式实证问题统一在：

[
\mathcal X={x\in\mathbb R^n:\mathbf 1^\top x=1,\ 0\le x_i\le x_{\max}}
]

上求解。

因此，solver 必须显式支持：

* simplex equality constraint；
* nonnegativity；
* upper box constraints；
  -（主模型中）band 约束的软实现。

### 2.4 数值稳定优先原则

求解器首先必须保证：

* 可行；
* 稳定；
* 可解释；
* 输出足够多的诊断量；

然后才追求速度。

---

## 3. 统一问题表述

### 3.1 三类模型对象

统一求解器需覆盖以下三类问题：

#### Model A: CtR-only

[
\min_{x\in\mathcal X} D_R(x;b)
]

#### Model B: CtB-only

[
\min_{x\in\mathcal X} D_B(x)
]

#### Model C: 主模型（概念型）

[
\min_{x\in\mathcal X}
D_R(x;b)+\eta|x-x_{t-1}|_2^2+\gamma|x|_2^2
\quad
\text{s.t.}\quad D_B(x)\le \delta
]

#### Model C': 主模型（实现型）

[
J(x)=D_R(x;b)+\eta|x-x_{t-1}|_2^2+\gamma|x|*2^2+\frac{\rho}{2}(D_B(x)-\delta)*+^2
]

### 3.2 统一 closure 原则

在代码层，三类模型都必须写成：

* `objective(x, state) -> scalar`
* `gradient(x, state) -> ndarray`
* `hessian(x, state) -> ndarray or linear operator`
* `constraints(state) -> feasible set metadata`

其中 `state` 统一包含：

* `date_t`
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
* `mode` in {`ctr_only`, `ctb_only`, `rb_ctb_band`}

---

## 4. 统一接口（冻结）

### 4.1 顶层接口

```python
solve_portfolio(state: SolverState, options: SolverOptions) -> SolverResult
```

### 4.2 `SolverState` 必须包含的字段

* `date_t`: 当前调仓日期
* `asset_list`: 资产列表
* `cov_t`: 协方差矩阵 V_t
* `corr_t`: 相关矩阵 C_t
* `sigma_t`: 资产波动率向量
* `x_prev`: 上一期漂移后权重
* `budget`: 风险预算向量 b
* `delta_band`: CtB band 阈值
* `eta_smooth`: 平滑参数 η
* `gamma_l2`: L2 稳定参数 γ
* `rho_penalty`: hinge penalty 参数 ρ
* `x_max`: 单资产权重上限
* `mode`: `ctr_only` / `ctb_only` / `rb_ctb_band`

### 4.3 `SolverOptions` 必须包含的字段

* `max_iter_pg`: 梯度阶段最大迭代数
* `max_iter_newton`: 牛顿阶段最大迭代数
* `tol_grad`: 梯度范数阈值
* `tol_obj`: 目标改变量阈值
* `tol_feas`: 可行性阈值
* `tol_kkt`: KKT 残差阈值
* `switch_grad_norm`: 从梯度阶段切换到牛顿阶段的阈值
* `step_init`: 初始步长
* `step_decay`: 回溯线搜索衰减系数
* `newton_damping_init`: 初始阻尼参数
* `newton_damping_decay`: 牛顿阻尼衰减规则
* `fallback_to_pg`: 是否允许失败后退回梯度法
* `store_trace`: 是否保存迭代轨迹

### 4.4 `SolverResult` 必须包含的字段

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
* `grad_norm_final`
* `kkt_residual_final`
* `band_active`
* `constraint_violation`
* `solver_status`
* `trace`（可选）

---

## 5. 初始化规则

### 5.1 默认初值

统一求解器的默认初值按以下优先级产生：

1. 若存在 `x_prev` 且可行，则以 `x_prev` 为初值；
2. 若 `x_prev` 不可行，则投影到可行域作为初值；
3. 若无 `x_prev`，使用等权：
   [
   x^{(0)}=\frac{1}{n}\mathbf 1
   ]
4. 若等权违反上限（例如 (1/n>x_{\max})），则使用 capped-simplex projection 生成可行初值。

### 5.2 warm start 规则

* 在滚动回测中，主模型和两类基准都优先使用上期最优权重的漂移后版本作为 warm start；
* 对 `rb_ctb_band`，若 warm start 带来明显 band violation，不直接拒绝，而是在梯度阶段中自动修正。

### 5.3 不允许的初始化

* 不允许随机初始化作为主线方案；
* 不允许对不同策略使用不同类型的初值而不报告。

---

## 6. 阶段 I：Projected Gradient（冻结）

### 6.1 使命

Projected Gradient 阶段负责：

* 在可行域内稳定下降；
* 快速降低目标值和一阶残差；
* 在进入牛顿阶段前稳定 active set；
* 避免直接牛顿造成的边界穿越和数值不稳定。

### 6.2 迭代格式

[
x^{(k+1)}=\Pi_{\mathcal X}\bigl(x^{(k)}-\alpha_k g(x^{(k)})\bigr)
]

其中：

* (g(x)=\nabla J(x))
* (\Pi_{\mathcal X}(\cdot)) 是对可行域的投影
* (\alpha_k) 通过回溯线搜索选取

### 6.3 步长规则

采用 backtracking line search：

1. 从 `step_init` 开始；
2. 若新点不可行，则先做投影；
3. 若目标下降不足，则令：
   [
   \alpha \leftarrow \beta \alpha,
   \qquad \beta \in (0,1)
   ]
4. 直到满足下降准则或达到最小步长。

### 6.4 目标下降准则

建议采用 Armijo 型或其投影变体：

[
J(x^{new})\le J(x^{old})-c\alpha|g(x^{old})|_2^2
]

其中常数 (c) 由实现层指定。

### 6.5 梯度阶段的终止条件

若满足以下任一条件，则结束梯度阶段：

1. (|g(x^{(k)})|_2 \le \texttt{switch_grad_norm})
2. 目标改变量低于 `tol_obj`
3. 迭代达到 `max_iter_pg`
4. active set 在连续若干轮中不再变化（可选增强规则）

### 6.6 梯度阶段输出

梯度阶段必须输出：

* 当前权重
* 当前目标值
* 当前梯度范数
* 当前 active set
* 当前 band violation
* 当前 `dr`, `db`

---

## 7. 阶段 II：Active-set Damped Newton（冻结）

### 7.1 使命

牛顿阶段负责：

* 在局部邻域内快速收敛；
* 更精确满足 stationarity / KKT 条件；
* 在 active set 稳定后提升精度。

### 7.2 为什么不是直接牛顿

根据笔记主线，直接使用牛顿法可能：

* 跳出定义域；
* 穿越边界；
* 改变应保持的可行结构；
* 在 CtB 或带约束的目标上引发不稳定。

因此牛顿阶段只允许在梯度阶段收敛到一定邻域后启动。

### 7.3 free set / active set

定义：

* `active_lower`: 当前在下界 (x_i=0) 的变量
* `active_upper`: 当前在上界 (x_i=x_{\max}) 的变量
* `free_set`: 其余变量

牛顿方向仅在 free variables 上构造。

### 7.4 牛顿方向

设自由变量子空间上的梯度和 Hessian 分别为 (g_F, H_F)，则阻尼牛顿方向定义为：

[
d_F = -(H_F + \mu_k I)^{-1} g_F
]

其中：

* (\mu_k\ge 0) 为阻尼参数；
* 在边界变量上方向固定为 0；
* 若 `H_F` 非正定或病态，则自动增大阻尼。

### 7.5 牛顿步长与可行性维护

牛顿更新形式：

[
x^{(k+1)} = x^{(k)} + \tau_k d_k
]

其中 (\tau_k) 通过可行线搜索确定，要求：

* 不突破下界和上界；
* 不破坏预算约束；
* 保证目标下降；
* band 罚项不出现数值爆炸。

### 7.6 牛顿阶段终止条件

若满足以下全部或主要条件，则认为牛顿阶段收敛：

1. (|g(x^{(k)})|_2 \le \texttt{tol_grad})
2. KKT 残差 (\le \texttt{tol_kkt})
3. 可行性违反 (\le \texttt{tol_feas})
4. 目标变化 (\le \texttt{tol_obj})

### 7.7 牛顿失败时的处理

若出现以下情形：

* Hessian 严重病态；
* 连续若干步步长极小；
* 目标不下降；
* KKT 残差震荡；

则：

1. 增大阻尼 (\mu_k)；
2. 若仍失败，退回梯度阶段若干步；
3. 若 `fallback_to_pg=True`，则以投影梯度作为保底收敛器。

---

## 8. 可行域投影与约束处理

### 8.1 预算约束与 box 约束

求解器必须支持将任意候选向量投影到 capped simplex：

[
{x:\mathbf 1^\top x = 1,\ 0\le x_i\le x_{\max}}
]

### 8.2 投影器要求

必须实现统一函数：

```python
project_to_capped_simplex(v, x_max) -> x_proj
```

要求：

* 输出严格可行；
* 数值稳定；
* 允许用于初始化、梯度阶段和牛顿失败回退。

### 8.3 band 约束处理

正文主模型是硬约束：
[
D_B(x)\le \delta
]

代码实现中采用 hinge penalty：
[
\frac{\rho}{2}(D_B(x)-\delta)_+^2
]

因此 solver 层对 band 的处理规则为：

* 在概念层视作约束；
* 在实现层视作目标中的 penalty term；
* 诊断层必须显式输出 `band_active` 与 `constraint_violation`。

---

## 9. 三类模型如何共享统一求解器

### 9.1 统一模式开关

通过 `state.mode` 决定 objective closure：

#### mode = `ctr_only`

[
J(x)=D_R(x;b)
]
或在实现中允许加入极小 (\gamma|x|_2^2) 仅作数值稳定。

#### mode = `ctb_only`

[
J(x)=D_B(x)
]
或在实现中允许加入极小 (\gamma|x|_2^2)。

#### mode = `rb_ctb_band`

[
J(x)=D_R(x;b)+\eta|x-x_{t-1}|_2^2+\gamma|x|*2^2+\frac{\rho}{2}(D_B(x)-\delta)*+^2
]

### 9.2 统一输出

无论是哪种 mode，SolverResult 都必须输出相同字段：

* `x_opt`
* `ctr`
* `ctb`
* `dr`
* `db`
* `objective_value`
* `solver_status`
* `diagnostics`

这样后续导出和对比才不会分裂成三套格式。

---

## 10. 诊断量（冻结）

每次求解必须输出以下诊断量，写入 `diagnostics.json` 或对应表格：

### 10.1 基础诊断

* `converged`
* `solver_status`
* `iterations_pg`
* `iterations_newton`
* `objective_value_final`
* `grad_norm_final`
* `kkt_residual_final`
* `constraint_violation_final`

### 10.2 结构诊断

* `band_active`
* `db_margin = db - delta_band`
* `active_lower_count`
* `active_upper_count`
* `free_set_count`

### 10.3 稳定性诊断

* `step_rejections_pg`
* `step_rejections_newton`
* `min_step_size_used`
* `max_damping_used`
* `fallback_triggered`

### 10.4 结果诊断

* `dr`
* `db`
* `obj_terms`

  * `dr_term`
  * `smooth_term`
  * `l2_term`
  * `band_penalty`

---

## 11. 统一伪代码（冻结）

```text
Input:
    state = {cov_t, corr_t, sigma_t, x_prev, budget, delta_band,
             eta_smooth, gamma_l2, rho_penalty, x_max, mode, ...}
    options = {max_iter_pg, max_iter_newton, tol_grad, tol_obj,
               tol_feas, tol_kkt, switch_grad_norm, step_init,
               step_decay, newton_damping_init, ...}

Output:
    SolverResult

Procedure SOLVE_PORTFOLIO(state, options):

1. Build objective closure J(x; state)
2. Build gradient closure g(x; state)
3. Build Hessian closure H(x; state) or Hessian approximation
4. Initialize x0:
       if x_prev exists and feasible:
           x <- x_prev
       else:
           x <- project_to_capped_simplex(equal_weight, x_max)
5. Evaluate diagnostics at x
6. If already converged:
       return result

Stage I: Projected Gradient
7. for k in 1..max_iter_pg:
       grad <- g(x)
       if norm(grad) <= switch_grad_norm:
           break
       alpha <- step_init
       repeat:
           x_candidate <- project_to_capped_simplex(x - alpha * grad, x_max)
           if J(x_candidate) <= sufficient_decrease_condition:
               accept
           else:
               alpha <- step_decay * alpha
           if alpha too small:
               mark line-search warning
               accept projected candidate or break
       x <- x_candidate
       update diagnostics and trace
       if objective change <= tol_obj:
           break

Stage II: Active-set Damped Newton
8. identify active_lower, active_upper, free_set from x
9. for m in 1..max_iter_newton:
       grad <- g(x)
       if norm(grad) <= tol_grad and KKT(x) <= tol_kkt and feasibility(x) <= tol_feas:
           mark converged and break
       build free-set gradient g_F
       build free-set Hessian H_F
       mu <- current damping
       repeat:
           solve d_F = -(H_F + mu I)^(-1) g_F
           expand d to full dimension with zeros on active coordinates
           choose feasible step tau by line search / fraction-to-boundary rule
           x_candidate <- project_to_capped_simplex(x + tau * d, x_max)
           if J(x_candidate) decreases and stability tests pass:
               accept
           else:
               increase damping mu and/or reduce tau
           if damping too large or tau too small:
               trigger Newton failure
               break
       if Newton failure:
           if fallback_to_pg:
               do several projected-gradient recovery steps
               continue or terminate with warning
           else:
               terminate with warning
       x <- x_candidate
       update active set, diagnostics, trace

10. Compute final ctr, ctb, dr, db, obj_terms
11. Compute final KKT residual and feasibility diagnostics
12. Return SolverResult
```

---

## 12. 各模式的 objective/gradient/Hessian 契约

### 12.1 CtR-only

#### Objective

[
J_{CtR}(x)=D_R(x;b)
]

#### Gradient

必须实现 `grad_dr(x, state)`。

#### Hessian

优先实现精确 Hessian；若实现成本过高，可采用：

* Gauss–Newton 近似；
* 数值 Hessian（仅调试/验证，不用于正式主线）；
* BFGS 近似（仅在实现层作为备选，不进入正文主算法主线）。

### 12.2 CtB-only

#### Objective

[
J_{CtB}(x)=D_B(x)
]

#### Gradient

必须实现 `grad_db(x, state)`。

#### Hessian

允许使用：

* 精确 Hessian；
* 稳定的近似 Hessian；
* 必须在文档中说明近似方式。

### 12.3 主模型

#### Objective

[
J(x)=D_R(x;b)+\eta|x-x_{t-1}|_2^2+\gamma|x|*2^2+\frac{\rho}{2}(D_B(x)-\delta)*+^2
]

#### Gradient

必须实现：

```python
grad_total = grad_dr + grad_smooth + grad_l2 + grad_band
```

其中：

* `grad_dr` 对应 (D_R)
* `grad_smooth = 2\eta (x-x_{t-1})`
* `grad_l2 = 2\gamma x`
* `grad_band = 0` if (D_B(x)\le \delta), else
  [
  \rho (D_B(x)-\delta)\nabla D_B(x)
  ]

#### Hessian

必须支持：

* 至少对 `grad_smooth` 和 `grad_l2` 的 Hessian 精确实现；
* `D_R` 和 band 项部分允许精确或近似 Hessian；
* 文档中必须注明实现策略。

---

## 13. 失败回退与工程保底

### 13.1 允许的 fallback

若统一求解器在某个日期/某个模式下失败，允许以下 fallback：

1. **PG-only fallback**
   继续使用投影梯度直到满足较宽松收敛条件；

2. **warm-start adjusted restart**
   使用更强的稳定项或更接近等权的初值重新启动；

3. **baseline safe solution**
   若主模型失败，可临时回退到 ERC 或上一期权重，但必须记录为异常日期。

### 13.2 不允许的 fallback

* 静默切换到完全不同的优化问题；
* 静默去掉 band；
* 静默修改预算向量；
* 静默放宽可行域。

### 13.3 失败记录要求

所有 fallback 都必须记录到：

* `diagnostics.json`
* `analysis_pack.json`
* `run_manifest.json`

---

## 14. 与代码模块的映射

### 14.1 模块建议

```text
src/
  solvers/
    core.py
    project.py
    pg.py
    newton.py
    diagnostics.py
  objectives/
    ctr.py
    ctb.py
    main_model.py
```

### 14.2 职责划分

* `core.py`: 顶层统一接口 `solve_portfolio`
* `project.py`: capped-simplex 投影
* `pg.py`: projected gradient
* `newton.py`: active-set damped Newton
* `diagnostics.py`: KKT、残差、约束与收敛统计
* `objectives/ctr.py`: CtR-only objective/grad/hess
* `objectives/ctb.py`: CtB-only objective/grad/hess
* `objectives/main_model.py`: 主模型 objective/grad/hess

---

## 15. 与论文章节的映射

### 第 3 章

* 用本文件中的两阶段框架说明 CtR 求解思路；
* 不给出全部工程细节，只讲思想和必要公式。

### 第 4 章

* 说明 CtB 问题由于结构更非线性，因此更需要先梯度后牛顿；
* 不展开代码接口。

### 第 5 章

* 不讨论 solver。

### 第 6 章

* 本文件是第 6 章的正文蓝本；
* 可直接转化为“统一求解器与数值实现”章节内容。

---

## 16. 冻结结论

从本文件起，本项目的求解器层固定如下：

1. 使用一个统一 solver family；
2. 三类模型通过 objective closure 区分，而不是通过三套独立 solver 区分；
3. 正式主算法固定为：

   * 阶段 I：Projected Gradient
   * 阶段 II：Active-set Damped Newton
4. 所有模型统一在 capped simplex 上求解；
5. 所有求解过程都必须输出完整诊断量；
6. fallback 只能在本文件允许的范围内进行；
7. 代码与论文后续必须服从本契约。

```

这份 `07_solver_contract.md` 已经把你后面最关键的东西都钉住了：

- 不是三套 solver，而是一个统一求解器家族；
- 明确体现了笔记中的“先梯度、后牛顿”；
- 同时兼容 CtR-only、CtB-only 和主模型；
- 把工程实现最容易漂移的地方（投影、active-set、fallback、诊断）都写死了。fileciteturn19file0 fileciteturn19file2

下一步最合适的是：基于 `05_data_contract.md`、`06_model_contract.md` 和这份 `07_solver_contract.md`，继续冻结 **`08_result_contract.md`**，把每次实验必须输出的图表、表格、CSV/JSON、诊断文件和命名规则一次性写死。

```
