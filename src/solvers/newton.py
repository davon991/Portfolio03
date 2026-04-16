from __future__ import annotations

from typing import Callable

import numpy as np

from src.solvers.project import project_to_capped_simplex


Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]


def finite_diff_hessian(x: np.ndarray, gradient: Gradient, eps: float) -> np.ndarray:
    n = len(x)
    h = np.zeros((n, n), dtype=float)
    g0 = gradient(x)
    for i in range(n):
        xp = x.copy()
        xp[i] += eps
        gp = gradient(xp)
        h[:, i] = (gp - g0) / eps
    return 0.5 * (h + h.T)


def active_set_damped_newton(
    x0: np.ndarray,
    objective: Objective,
    gradient: Gradient,
    x_max: float,
    max_iter: int,
    tol_grad: float,
    tol_obj: float,
    fd_eps: float,
    damping_init: float,
    damping_mult: float,
    max_damping: float,
    step_decay: float,
    min_step: float,
    store_trace: bool = True,
) -> dict:
    x_max = float(x_max)
    max_iter = int(max_iter)
    tol_grad = float(tol_grad)
    tol_obj = float(tol_obj)
    fd_eps = float(fd_eps)
    damping_init = float(damping_init)
    damping_mult = float(damping_mult)
    max_damping = float(max_damping)
    step_decay = float(step_decay)
    min_step = float(min_step)

    x = np.asarray(x0, dtype=float).copy()
    obj = float(objective(x))
    trace = []
    step_rejections = 0
    fallback_triggered = False
    solver_status = "ok"

    for k in range(1, max_iter + 1):
        g = gradient(x)
        grad_norm = float(np.linalg.norm(g))
        if grad_norm <= tol_grad:
            break

        free = (x > 1e-10) & (x < x_max - 1e-10)
        if int(free.sum()) <= 1:
            solver_status = "free_set_too_small"
            break

        idx = np.where(free)[0]
        H = finite_diff_hessian(x, gradient, fd_eps)
        Hf = H[np.ix_(idx, idx)]
        gf = g[idx]

        # Impose approximate sum-to-zero direction in free subspace.
        P = np.eye(len(idx)) - np.ones((len(idx), len(idx)), dtype=float) / len(idx)
        Ht = P @ Hf @ P
        gt = P @ gf

        mu = damping_init
        accepted = False
        x_new = x.copy()
        obj_new = obj

        while mu <= max_damping:
            try:
                d_free = -np.linalg.solve(Ht + mu * np.eye(len(idx)), gt)
            except np.linalg.LinAlgError:
                mu *= damping_mult
                continue

            d = np.zeros_like(x, dtype=float)
            d[idx] = d_free
            d[idx] -= np.mean(d[idx])

            step = 1.0
            while step >= min_step:
                candidate = project_to_capped_simplex(x + step * d, x_max)
                candidate_obj = float(objective(candidate))
                if candidate_obj <= obj - 1e-4 * step * grad_norm * grad_norm:
                    x_new = candidate
                    obj_new = candidate_obj
                    accepted = True
                    break
                step *= step_decay
                step_rejections += 1

            if accepted:
                break
            mu *= damping_mult

        if not accepted:
            fallback_triggered = True
            solver_status = "newton_fallback"
            break

        x = x_new
        prev_obj = obj
        obj = obj_new

        if store_trace:
            trace.append(
                {
                    "phase": "newton",
                    "iter": k,
                    "objective_value": float(obj),
                    "grad_norm": float(np.linalg.norm(gradient(x))),
                    "step_size": float(step),
                    "damping": float(mu),
                }
            )

        if abs(prev_obj - obj) <= tol_obj:
            break

    return {
        "x": x,
        "objective_value": float(obj),
        "trace": trace,
        "step_rejections_newton": int(step_rejections),
        "fallback_triggered": bool(fallback_triggered),
        "solver_status": solver_status,
    }