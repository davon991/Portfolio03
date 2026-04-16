from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src.utils import ensure_dir, safe_to_parquet, save_json


def monthly_rebalance_dates(returns_df: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.to_datetime(returns_df["date"])
    month_end_idx = dates.groupby([dates.dt.year, dates.dt.month]).idxmax()
    return pd.DatetimeIndex(dates.loc[month_end_idx].sort_values().values)


def estimate_covariance(window: pd.DataFrame, method: str = "ledoit_wolf") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = window.to_numpy(dtype=float)
    if method == "ledoit_wolf":
        lw = LedoitWolf().fit(arr)
        cov = lw.covariance_
    elif method == "sample":
        cov = np.cov(arr, rowvar=False, ddof=1)
    else:
        raise ValueError(f"Unsupported covariance method: {method}")
    sigma = np.sqrt(np.diag(cov))
    sigma_safe = np.where(sigma <= 1e-12, 1e-12, sigma)
    corr = cov / np.outer(sigma_safe, sigma_safe)
    corr = np.clip(corr, -1.0, 1.0)
    return cov, corr, sigma


def build_risk_panels(returns_csv: str | Path, cfg: dict) -> dict[str, str]:
    returns_df = pd.read_csv(returns_csv)
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    tickers = cfg["data"]["tickers"]
    window_L = int(cfg["experiment"]["window_L"])
    min_periods = int(cfg["risk"].get("min_periods", min(window_L, 252)))
    method = cfg["risk"]["method"]

    rebalance_dates = monthly_rebalance_dates(returns_df)
    cov_rows = []
    corr_rows = []
    vol_rows = []

    for dt in rebalance_dates:
        loc = returns_df.index[returns_df["date"] == dt]
        if len(loc) == 0:
            continue
        idx = int(loc[0])
        if idx < min_periods:
            continue
        start = max(0, idx - window_L + 1)
        window = returns_df.iloc[start: idx + 1][tickers]
        cov, corr, sigma = estimate_covariance(window, method=method)
        for i, a in enumerate(tickers):
            vol_rows.append({"date": dt, "asset": a, "sigma": float(sigma[i])})
            for j, b in enumerate(tickers):
                cov_rows.append({"date": dt, "asset_i": a, "asset_j": b, "cov": float(cov[i, j])})
                corr_rows.append({"date": dt, "asset_i": a, "asset_j": b, "corr": float(corr[i, j])})

    risk_dir = ensure_dir("data/risk")
    cov_df = pd.DataFrame(cov_rows)
    corr_df = pd.DataFrame(corr_rows)
    vol_df = pd.DataFrame(vol_rows)

    cov_path = safe_to_parquet(cov_df, risk_dir / "cov_panel.parquet")
    corr_path = safe_to_parquet(corr_df, risk_dir / "corr_panel.parquet")
    vol_csv = risk_dir / "vol_panel.csv"
    vol_df.to_csv(vol_csv, index=False)

    save_json(
        risk_dir / "risk_estimation_meta.json",
        {
            "method": method,
            "window_L": window_L,
            "min_periods": min_periods,
            "n_rebalance_dates": int(vol_df["date"].nunique()) if not vol_df.empty else 0,
        },
    )

    return {
        "cov_panel": str(cov_path),
        "corr_panel": str(corr_path),
        "vol_panel": str(vol_csv),
    }