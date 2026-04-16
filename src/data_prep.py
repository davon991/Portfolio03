from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_dir, safe_to_parquet, save_json


@dataclass
class DataArtifacts:
    prices_path: str
    returns_path: str
    calendar_path: str
    asset_summary_path: str


def _generate_synthetic_prices(cfg: dict[str, Any]) -> pd.DataFrame:
    tickers = cfg["data"]["tickers"]
    start = cfg["data"]["start"]
    end = cfg["data"]["end"]
    seed = int(cfg["data"].get("synthetic_seed", 20260416))
    dates = pd.bdate_range(start=start, end=end)
    n = len(tickers)
    rng = np.random.default_rng(seed)

    # Structured correlation: equities clustered, rates clustered, real assets mixed.
    vols = np.array([0.18, 0.17, 0.22, 0.07, 0.11, 0.07, 0.09, 0.16, 0.14, 0.19]) / np.sqrt(252)
    corr = np.full((n, n), 0.12)
    np.fill_diagonal(corr, 1.0)
    equity = [0, 1, 2]
    rates = [3, 4, 5]
    credit = [6]
    real = [7, 8, 9]
    for grp, rho in [(equity, 0.75), (rates, 0.80), (real, 0.45)]:
        for i in grp:
            for j in grp:
                if i != j:
                    corr[i, j] = rho
    # Cross-cluster structure.
    for i in equity:
        for j in rates:
            corr[i, j] = corr[j, i] = -0.20
    for i in equity:
        corr[i, 6] = corr[6, i] = 0.45
    for i in equity:
        for j in real:
            corr[i, j] = corr[j, i] = 0.25
    corr[7, 8] = corr[8, 7] = 0.10
    corr[7, 9] = corr[9, 7] = 0.35
    corr[8, 9] = corr[9, 8] = 0.15

    cov = np.outer(vols, vols) * corr
    mean_daily = np.array([0.08, 0.07, 0.09, 0.03, 0.035, 0.03, 0.045, 0.06, 0.05, 0.055]) / 252
    rets = rng.multivariate_normal(mean_daily, cov, size=len(dates))
    prices = 100 * np.exp(np.cumsum(np.log1p(rets), axis=0))

    df = pd.DataFrame(prices, columns=tickers)
    df.insert(0, "date", dates)
    return df


def _load_local_prices(cfg: dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["data"]["raw_prices_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Local price file not found: {path}. Please provide a wide CSV with columns date + tickers."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    tickers = cfg["data"]["tickers"]
    missing = [c for c in tickers if c not in df.columns]
    if missing:
        raise ValueError(f"Price file is missing ticker columns: {missing}")
    return df[["date", *tickers]].sort_values("date").reset_index(drop=True)


def prepare_returns(cfg: dict[str, Any]) -> dict[str, str]:
    source = cfg["data"]["source"]
    processed_dir = ensure_dir("data/processed")
    categories = cfg["data"].get("categories", {})

    if source == "synthetic":
        prices = _generate_synthetic_prices(cfg)
    elif source == "local_csv":
        prices = _load_local_prices(cfg)
    else:
        raise ValueError(f"Unsupported data.source: {source}")

    tickers = cfg["data"]["tickers"]
    prices = prices.dropna(subset=tickers, how="all").copy()
    prices[tickers] = prices[tickers].ffill()
    prices = prices.dropna(subset=tickers, how="any").reset_index(drop=True)

    returns = prices[["date", *tickers]].copy()
    returns[tickers] = returns[tickers].pct_change()
    returns = returns.dropna().reset_index(drop=True)

    prices_csv = processed_dir / "prices_daily_adj.csv"
    returns_csv = processed_dir / "returns_daily.csv"
    calendar_csv = processed_dir / "calendar_master.csv"
    asset_summary_csv = processed_dir / "asset_summary.csv"

    prices.to_csv(prices_csv, index=False)
    returns.to_csv(returns_csv, index=False)
    pd.DataFrame({"date": returns["date"]}).to_csv(calendar_csv, index=False)

    summary_rows = []
    for t in tickers:
        s = returns[t].dropna()
        summary_rows.append(
            {
                "asset": t,
                "category": categories.get(t, "unknown"),
                "first_valid_date": str(returns.loc[s.index.min(), "date"].date()) if len(s) else None,
                "last_valid_date": str(returns.loc[s.index.max(), "date"].date()) if len(s) else None,
                "obs_count": int(len(s)),
                "mean_daily_return": float(s.mean()),
                "daily_vol": float(s.std(ddof=1)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(asset_summary_csv, index=False)

    # Optional parquet mirrors.
    safe_to_parquet(prices, processed_dir / "prices_daily_adj.parquet")

    save_json(
        processed_dir / "data_audit_report.json",
        {
            "source": source,
            "start": str(prices["date"].min().date()),
            "end": str(prices["date"].max().date()),
            "n_assets": len(tickers),
            "n_obs_returns": int(len(returns)),
            "tickers": tickers,
        },
    )

    return {
        "prices_csv": str(prices_csv),
        "returns_csv": str(returns_csv),
        "calendar_csv": str(calendar_csv),
        "asset_summary_csv": str(asset_summary_csv),
    }