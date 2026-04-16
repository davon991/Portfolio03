from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.diagnostics import compute_kkt_residual
from src.metrics import ctb, ctr, objective_terms
from src.solvers.newton import active_set_damped_newton
from src.solvers.pg import projected_gradient_descent
from src.solvers.project import project_to_capped_simplex


@dataclass
class SolverState:
    date_t: str
    cov_t: np.ndarray
    x_prev: np.ndarray | None
    budget: np.ndarray
    delta_band: float
    eta_smooth: float
    gamma_l2: float
    rho_penalty: float
    x_max: float
    mode: str

    def __post_init__(self) -> None:
        self.date_t = str(self.date_t)
        self.cov_t = np.asarray(self.cov_t, dtype=float)
        self.x_prev = None if self.x_prev is None else np.asarray(self.x_prev, dtype=float)
        self.budget = np.asarray(self.budget, dtype=float)
        self.delta_band = float(self.delta_band)
        self.eta_smooth = float(self.eta_smooth)
        self.gamma_l2 = float(self.gamma_l2)
        self.rho_penalty = float(self.rho_penalty)
        self.x_max = float(self.x_max)
        self.mode = str(self.mode)


@dataclass
class SolverOptions:
    max_iter_pg: int
    max_iter_newton: int
    tol_grad: float
    tol_obj: float
    tol_feas: float
    tol_kkt: float
    switch_grad_norm: float
    step_init: float
    step_decay: float
    min_step: float
    newton_damping_init: float
    newton_damping_mult: float
    max_damping: float
    fd_eps: float
    fallback_to_pg: bool
    store_trace: bool

    def __post_init__(self) -> None:
        self.max_iter_pg = int(self.max_iter_pg)
        self.max_iter_newton = int(self.max_iter_newton)
        self.tol_grad = float(self.tol_grad)
        self.tol_obj = float(self.tol_obj)
        self.tol_feas = float(self.tol_feas)
        self.tol_kkt = float(self.tol_kkt)
        self.switch_grad_norm = float(self.switch_grad_norm)
        self.step_init = float(self.step_init)
        self.step_decay = float(self.step_decay)
        self.min_step = float(self.min_step)
        self.newton_damping_init = float(self.newton_damping_init)
        self.newton_damping_mult = float(self.newton_damping_mult)
        self.max_damping = float(self.max_damping)
        self.fd_eps = float(self.fd_eps)
        self.fallback_to_pg = bool(self.fallback_to_pg)
        self.store_trace = bool(self.store_trace)


def _build_objective_and_gradient(
    state: SolverState,
) -> tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    x_prev = state.x_prev
    cov = state.cov_t
    budget = state.budget

    def objective(x: np.ndarray) -> float:
        terms = objective_terms(
            x=x,
            cov=cov,
            budget=budget,
            x_prev=x_prev,
            delta_band=state.delta_band,
            eta_smooth=state.eta_smooth,
            gamma_l2=state.gamma_l2,
            rho_penalty=state.rho_penalty,
            mode=state.mode,
        )
        return float(terms["obj_total"])

    def gradient(x: np.ndarray) -> np.ndarray:
        # 9A: robust finite-difference gradient for all three modes.
        eps = 1e-6
        g = np.zeros_like(x, dtype=float)
        f0 = objective(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            xp = project_to_capped_simplex(xp, state.x_max)
            g[i] = (objective(xp) - f0) / eps
        return g

    return objective, gradient


def solve_portfolio(state: SolverState, options: SolverOptions) -> dict:
    n = len(state.budget)

    if state.x_prev is not None:
        x0 = project_to_capped_simplex(state.x_prev, state.x_max)
    else:
        x0 = project_to_capped_simplex(np.full(n, 1.0 / n, dtype=float), state.x_max)

    objective, gradient = _build_objective_and_gradient(state)

    pg_res = projected_gradient_descent(
        x0=x0,
        objective=objective,
        gradient=gradient,
        x_max=state.x_max,
        max_iter=options.max_iter_pg,
        step_init=options.step_init,
        step_decay=options.step_decay,
        min_step=options.min_step,
        switch_grad_norm=options.switch_grad_norm,
        tol_obj=options.tol_obj,
        store_trace=options.store_trace,
    )

    x_pg = pg_res["x"]

    newton_res = active_set_damped_newton(
        x0=x_pg,
        objective=objective,
        gradient=gradient,
        x_max=state.x_max,
        max_iter=options.max_iter_newton,
        tol_grad=options.tol_grad,
        tol_obj=options.tol_obj,
        fd_eps=options.fd_eps,
        damping_init=options.newton_damping_init,
        damping_mult=options.newton_damping_mult,
        max_damping=options.max_damping,
        step_decay=options.step_decay,
        min_step=options.min_step,
        store_trace=options.store_trace,
    )

    x_final = newton_res["x"]
    if newton_res["fallback_triggered"] and options.fallback_to_pg:
        x_final = x_pg

    g_final = gradient(x_final)
    terms = objective_terms(
        x=x_final,
        cov=state.cov_t,
        budget=state.budget,
        x_prev=state.x_prev,
        delta_band=state.delta_band,
        eta_smooth=state.eta_smooth,
        gamma_l2=state.gamma_l2,
        rho_penalty=state.rho_penalty,
        mode=state.mode,
    )
    c_r = ctr(x_final, state.cov_t)
    c_b = ctb(x_final, state.cov_t)
    kkt = compute_kkt_residual(x_final, g_final)
    db_margin = float(terms["db"]) - state.delta_band

    trace = []
    if options.store_trace:
        trace.extend(pg_res["trace"])
        trace.extend(newton_res["trace"])

    return {
        "x_opt": x_final,
        "objective_value": float(terms["obj_total"]),
        "ctr": c_r,
        "ctb": c_b,
        "dr": float(terms["dr"]),
        "db": float(terms["db"]),
        "obj_terms": terms,
        "converged": bool(kkt <= options.tol_kkt),
        "iterations_pg": int(len([t for t in trace if t["phase"] == "pg"])),
        "iterations_newton": int(len([t for t in trace if t["phase"] == "newton"])),
        "grad_norm_final": float(np.linalg.norm(g_final)),
        "kkt_residual_final": float(kkt),
        "constraint_violation": float(max(0.0, db_margin)),
        "band_active": int(db_margin > 0),
        "db_margin": float(db_margin),
        "solver_status": newton_res["solver_status"],
        "trace": trace,
        "step_rejections_pg": int(pg_res["step_rejections_pg"]),
        "step_rejections_newton": int(newton_res["step_rejections_newton"]),
        "fallback_triggered": bool(newton_res["fallback_triggered"]),
    }