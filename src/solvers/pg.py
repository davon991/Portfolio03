from __future__ import annotations

from typing import Callable

import numpy as np

from src.solvers.project import project_to_capped_simplex


Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]


def projected_gradient_descent(
    x0: np.ndarray,
    objective: Objective,
    gradient: Gradient,
    x_max: float,
    max_iter: int,
    step_init: float,
    step_decay: float,
    min_step: float,
    switch_grad_norm: float,
    tol_obj: float,
    store_trace: bool = True,
) -> dict:
    x = x0.copy()
    trace = []
    obj_prev = objective(x)
    step_rejections = 0

    for k in range(1, max_iter + 1):
        g = gradient(x)
        grad_norm = float(np.linalg.norm(g))
        if store_trace:
            trace.append({
                "phase": "pg",
                "iter": k,
                "objective_value": obj_prev,
                "grad_norm": grad_norm,
                "step_size": 0.0,
                "damping": 0.0,
            })
        if grad_norm <= switch_grad_norm:
            break

        step = step_init
        accepted = False
        while step >= min_step:
            x_new = project_to_capped_simplex(x - step * g, x_max)
            obj_new = objective(x_new)
            if obj_new <= obj_prev - 1e-4 * step * grad_norm * grad_norm:
                accepted = True
                break
            step *= step_decay
            step_rejections += 1

        if not accepted:
            x_new = project_to_capped_simplex(x - min_step * g, x_max)
            obj_new = objective(x_new)

        if store_trace:
            trace.append({
                "phase": "pg",
                "iter": k,
                "objective_value": obj_new,
                "grad_norm": float(np.linalg.norm(gradient(x_new))),
                "step_size": step,
                "damping": 0.0,
            })

        if abs(obj_prev - obj_new) <= tol_obj:
            x = x_new
            obj_prev = obj_new
            break

        x = x_new
        obj_prev = obj_new

    return {
        "x": x,
        "objective_value": obj_prev,
        "trace": trace,
        "step_rejections_pg": step_rejections,
    }