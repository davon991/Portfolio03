from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_kkt_residual(x: np.ndarray, grad: np.ndarray, tol: float = 1e-8) -> float:
    # Practical 9A proxy: norm of projected gradient on simplex/box.
    mask_free = (x > tol) & (x < 1.0 - tol)
    if mask_free.any():
        free_grad = grad[mask_free] - np.mean(grad[mask_free])
        return float(np.linalg.norm(free_grad))
    return float(np.linalg.norm(grad - np.mean(grad)))


def aggregate_diagnostics(solver_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not solver_records:
        return {
            "solve_success_rate": 0.0,
            "fallback_rate": 0.0,
            "active_rate": 0.0,
            "kkt_residual_stats": {},
            "constraint_violation_stats": {},
            "band_violation_stats": {},
            "warning_dates": [],
            "failed_dates": [],
        }

    df = pd.DataFrame(solver_records)
    return {
        "solve_success_rate": float(df["converged"].mean()),
        "fallback_rate": float(df["fallback_triggered"].mean()) if "fallback_triggered" in df else 0.0,
        "active_rate": float(df["band_active"].mean()) if "band_active" in df else 0.0,
        "kkt_residual_stats": {
            "mean": float(df["kkt_residual_final"].mean()),
            "p95": float(df["kkt_residual_final"].quantile(0.95)),
            "max": float(df["kkt_residual_final"].max()),
        },
        "constraint_violation_stats": {
            "mean": float(df["constraint_violation"].mean()),
            "max": float(df["constraint_violation"].max()),
        },
        "band_violation_stats": {
            "mean": float(df["db_margin"].mean()) if "db_margin" in df else 0.0,
            "max": float(df["db_margin"].max()) if "db_margin" in df else 0.0,
        },
        "warning_dates": df.loc[df["solver_status"] != "ok", "date"].astype(str).tolist(),
        "failed_dates": df.loc[~df["converged"], "date"].astype(str).tolist(),
    }