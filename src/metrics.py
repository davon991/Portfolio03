from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def portfolio_var(x: np.ndarray, cov: np.ndarray) -> float:
    return float(x @ cov @ x)


def portfolio_vol(x: np.ndarray, cov: np.ndarray) -> float:
    return math.sqrt(max(portfolio_var(x, cov), 1e-16))


def ctr(x: np.ndarray, cov: np.ndarray) -> np.ndarray:
    denom = max(float(x @ cov @ x), 1e-16)
    raw = x * (cov @ x)
    return raw / denom


def ctb(x: np.ndarray, cov: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.maximum(np.diag(cov), 1e-16))
    pvol = portfolio_vol(x, cov)
    return (cov @ x) / (sigma * max(pvol, 1e-16))


def dr_value(x: np.ndarray, cov: np.ndarray, budget: np.ndarray) -> float:
    c = ctr(x, cov)
    return float(np.sum((c - budget) ** 2))


def db_value(x: np.ndarray, cov: np.ndarray) -> float:
    b = ctb(x, cov)
    m = float(np.mean(b))
    return float(np.sum((b - m) ** 2))


def objective_terms(
    x: np.ndarray,
    cov: np.ndarray,
    budget: np.ndarray,
    x_prev: np.ndarray | None,
    delta_band: float,
    eta_smooth: float,
    gamma_l2: float,
    rho_penalty: float,
    mode: str,
) -> dict[str, float]:
    dr = dr_value(x, cov, budget)
    db = db_value(x, cov)
    smooth = float(eta_smooth * np.sum((x - x_prev) ** 2)) if x_prev is not None else 0.0
    l2 = float(gamma_l2 * np.sum(x**2))
    band_penalty = 0.0

    if mode == "RB_CTB_BAND":
        band_penalty = 0.5 * rho_penalty * max(db - delta_band, 0.0) ** 2
        total = dr + smooth + l2 + band_penalty
    elif mode == "ERC":
        total = dr + l2
    elif mode == "CTB_ONLY":
        total = db + l2
    elif mode == "GMV":
        total = portfolio_var(x, cov) + l2
    elif mode == "EW":
        total = l2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "obj_total": float(total),
        "dr_term": float(dr),
        "smooth_term": float(smooth),
        "l2_term": float(l2),
        "band_penalty": float(band_penalty),
        "dr": float(dr),
        "db": float(db),
    }


def summarize_performance(perf_daily: pd.DataFrame, dr_db: pd.DataFrame, turnover_ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, g in perf_daily.groupby("strategy"):
        g = g.sort_values("date")
        rets = g["portfolio_return"].to_numpy(dtype=float)
        nav = g["nav"].to_numpy(dtype=float)
        ann_return = float((1.0 + rets).prod() ** (252 / max(len(rets), 1)) - 1.0)
        ann_vol = float(np.std(rets, ddof=1) * np.sqrt(252)) if len(rets) > 1 else 0.0
        sharpe = ann_return / ann_vol if ann_vol > 1e-12 else np.nan
        dd = g["drawdown"].to_numpy(dtype=float)
        max_dd = float(dd.min()) if len(dd) else 0.0

        tt = turnover_ts.loc[turnover_ts["strategy"] == strategy, "turnover"].astype(float)
        dr_s = dr_db.loc[dr_db["strategy"] == strategy, "dr"].astype(float)
        db_s = dr_db.loc[dr_db["strategy"] == strategy, "db"].astype(float)
        active = dr_db.loc[dr_db["strategy"] == strategy, "band_active"].astype(int)

        rows.append(
            {
                "strategy": strategy,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "turnover_mean": float(tt.mean()) if len(tt) else 0.0,
                "turnover_p95": float(tt.quantile(0.95)) if len(tt) else 0.0,
                "dr_mean": float(dr_s.mean()) if len(dr_s) else np.nan,
                "db_mean": float(db_s.mean()) if len(db_s) else np.nan,
                "active_rate": float(active.mean()) if len(active) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("strategy").reset_index(drop=True)


def nav_and_drawdown(returns: pd.Series) -> pd.DataFrame:
    nav = (1.0 + returns).cumprod()
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    return pd.DataFrame({"nav": nav, "drawdown": drawdown})