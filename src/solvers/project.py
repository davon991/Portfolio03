from __future__ import annotations

import numpy as np


def project_to_capped_simplex(v: np.ndarray, x_max: float, tol: float = 1e-12) -> np.ndarray:
    """Project vector v onto {x: sum x = 1, 0 <= x_i <= x_max}."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if x_max * n < 1.0 - 1e-12:
        raise ValueError("Infeasible capped simplex: x_max * n < 1.")

    lower = np.min(v - x_max)
    upper = np.max(v)
    for _ in range(200):
        lam = 0.5 * (lower + upper)
        x = np.clip(v - lam, 0.0, x_max)
        s = x.sum()
        if abs(s - 1.0) <= tol:
            return x
        if s > 1.0:
            lower = lam
        else:
            upper = lam
    x = np.clip(v - 0.5 * (lower + upper), 0.0, x_max)
    s = x.sum()
    if s <= tol:
        x = np.full(n, 1.0 / n)
        x = np.clip(x, 0.0, x_max)
        x = x / x.sum()
        return x
    return x / s