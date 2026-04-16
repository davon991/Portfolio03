from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.covariance import build_risk_panels, estimate_covariance, monthly_rebalance_dates
from src.diagnostics import aggregate_diagnostics
from src.metrics import nav_and_drawdown, summarize_performance
from src.solvers import SolverOptions, SolverState, solve_portfolio
from src.solvers.project import project_to_capped_simplex
from src.utils import ensure_dir, save_json, safe_to_parquet


def _budget_vector(n: int, budget_type: str) -> np.ndarray:
    if budget_type == "equal":
        return np.full(n, 1.0 / n)
    raise ValueError(f"Unsupported budget_type: {budget_type}")


def _gmv_weights(cov: np.ndarray, x_max: float) -> np.ndarray:
    n = cov.shape[0]
    inv = np.linalg.pinv(cov)
    one = np.ones(n)
    x = inv @ one
    x = x / np.sum(x)
    return project_to_capped_simplex(x, x_max)


def run_full_pipeline(cfg: dict[str, Any], run_dir: str | Path, data_artifacts: dict[str, str]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    panels_dir = ensure_dir(run_dir / "panels")
    summaries_dir = ensure_dir(run_dir / "summaries")
    figures_dir = ensure_dir(run_dir / "figures")
    tables_dir = ensure_dir(run_dir / "tables")
    logs_dir = ensure_dir(run_dir / "logs")

    returns_df = pd.read_csv(data_artifacts["returns_csv"])
    returns_df["date"] = pd.to_datetime(returns_df["date"])
    tickers = cfg["data"]["tickers"]
    strategies = cfg["experiment"]["strategies"]
    x_max = float(cfg["experiment"]["x_max"])
    budget = _budget_vector(len(tickers), cfg["model"]["budget_type"])
    window_L = int(cfg["experiment"]["window_L"])

    # Ensure risk panels are written for contract completeness.
    build_risk_panels(data_artifacts["returns_csv"], cfg)

    rebalance_dates = monthly_rebalance_dates(returns_df)
    options = SolverOptions(**cfg["solver"])

    weights_rows = []
    ctr_rows = []
    ctb_rows = []
    drdb_rows = []
    obj_rows = []
    turnover_rows = []
    perf_rows = []
    solver_records = []
    trace_rows = []

    prev_target = {s: None for s in strategies}
    prev_drifted = {s: None for s in strategies}

    valid_rebalance_dates = []
    for dt in rebalance_dates:
        idx = returns_df.index[returns_df["date"] == dt]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        if i < window_L - 1:
            continue
        valid_rebalance_dates.append(dt)

    for pos, dt in enumerate(valid_rebalance_dates[:-1]):
        next_dt = valid_rebalance_dates[pos + 1]
        idx = int(returns_df.index[returns_df["date"] == dt][0])
        start = idx - window_L + 1
        window = returns_df.iloc[start: idx + 1][tickers]
        cov, corr, sigma = estimate_covariance(window, method=cfg["risk"]["method"])

        # Next holding period (open interval after rebalance date until next rebalance date inclusive)
        hold_mask = (returns_df["date"] > dt) & (returns_df["date"] <= next_dt)
        hold_df = returns_df.loc[hold_mask, ["date", *tickers]].copy()
        if hold_df.empty:
            continue

        for strategy in strategies:
            if strategy == "EW":
                x = project_to_capped_simplex(np.full(len(tickers), 1.0 / len(tickers)), x_max)
                result = {
                    "x_opt": x,
                    "ctr": np.zeros(len(tickers)),
                    "ctb": np.zeros(len(tickers)),
                    "dr": np.nan,
                    "db": np.nan,
                    "obj_terms": {
                        "obj_total": 0.0,
                        "dr_term": np.nan,
                        "smooth_term": 0.0,
                        "l2_term": 0.0,
                        "band_penalty": 0.0,
                    },
                    "converged": True,
                    "iterations_pg": 0,
                    "iterations_newton": 0,
                    "grad_norm_final": 0.0,
                    "kkt_residual_final": 0.0,
                    "constraint_violation": 0.0,
                    "band_active": 0,
                    "db_margin": 0.0,
                    "solver_status": "ok",
                    "trace": [],
                    "step_rejections_pg": 0,
                    "step_rejections_newton": 0,
                    "fallback_triggered": False,
                }
            elif strategy == "GMV":
                x = _gmv_weights(cov, x_max)
                result = {
                    "x_opt": x,
                    "ctr": np.zeros(len(tickers)),
                    "ctb": np.zeros(len(tickers)),
                    "dr": np.nan,
                    "db": np.nan,
                    "obj_terms": {
                        "obj_total": float(x @ cov @ x),
                        "dr_term": np.nan,
                        "smooth_term": 0.0,
                        "l2_term": 0.0,
                        "band_penalty": 0.0,
                    },
                    "converged": True,
                    "iterations_pg": 0,
                    "iterations_newton": 0,
                    "grad_norm_final": 0.0,
                    "kkt_residual_final": 0.0,
                    "constraint_violation": 0.0,
                    "band_active": 0,
                    "db_margin": 0.0,
                    "solver_status": "ok",
                    "trace": [],
                    "step_rejections_pg": 0,
                    "step_rejections_newton": 0,
                    "fallback_triggered": False,
                }
            else:
                state = SolverState(
                    date_t=str(dt.date()),
                    cov_t=cov,
                    x_prev=prev_target[strategy],
                    budget=budget,
                    delta_band=float(cfg["model"]["delta_band"]),
                    eta_smooth=float(cfg["model"]["eta_smooth"]),
                    gamma_l2=float(cfg["model"]["gamma_l2"]),
                    rho_penalty=float(cfg["model"]["rho_penalty"]),
                    x_max=x_max,
                    mode=strategy,
                )
                result = solve_portfolio(state, options)
                x = result["x_opt"]

            # Portfolio returns over the holding period.
            period_rets = hold_df[tickers].to_numpy(dtype=float)
            port_rets = period_rets @ x
            navdd = nav_and_drawdown(pd.Series(port_rets, index=hold_df.index))
            for irow, row in hold_df.iterrows():
                perf_rows.append(
                    {
                        "date": row["date"],
                        "strategy": strategy,
                        "portfolio_return": float(period_rets[list(hold_df.index).index(irow)] @ x),
                        "nav": float(navdd.loc[irow, "nav"]),
                        "drawdown": float(navdd.loc[irow, "drawdown"]),
                    }
                )

            # Drifted weights at period end.
            gross = np.prod(1.0 + period_rets, axis=0)
            drifted = x * gross
            drifted = drifted / drifted.sum()
            turnover = 0.0
            if prev_drifted[strategy] is not None:
                turnover = 0.5 * float(np.sum(np.abs(x - prev_drifted[strategy])))
            prev_target[strategy] = x.copy()
            prev_drifted[strategy] = drifted.copy()

            for j, asset in enumerate(tickers):
                weights_rows.append({"date": dt, "strategy": strategy, "asset": asset, "weight": float(x[j])})
                if strategy not in ("EW", "GMV"):
                    ctr_rows.append({"date": dt, "strategy": strategy, "asset": asset, "ctr": float(result["ctr"][j])})
                    ctb_rows.append({"date": dt, "strategy": strategy, "asset": asset, "ctb": float(result["ctb"][j])})
                    for tr in result["trace"]:
                        trace_rows.append(
                            {
                                "date": dt,
                                "strategy": strategy,
                                **tr,
                                "kkt_residual": float(result["kkt_residual_final"]),
                                "constraint_violation": float(result["constraint_violation"]),
                                "band_active": int(result["band_active"]),
                            }
                        )

            if strategy not in ("EW", "GMV"):
                drdb_rows.append(
                    {
                        "date": dt,
                        "strategy": strategy,
                        "dr": float(result["dr"]),
                        "db": float(result["db"]),
                        "band_active": int(result["band_active"]),
                    }
                )
                obj_rows.append(
                    {
                        "date": dt,
                        "strategy": strategy,
                        **result["obj_terms"],
                    }
                )
                solver_records.append(
                    {
                        "date": str(dt.date()),
                        "strategy": strategy,
                        "converged": bool(result["converged"]),
                        "iterations_pg": int(result["iterations_pg"]),
                        "iterations_newton": int(result["iterations_newton"]),
                        "kkt_residual_final": float(result["kkt_residual_final"]),
                        "constraint_violation": float(result["constraint_violation"]),
                        "band_active": int(result["band_active"]),
                        "db_margin": float(result["db_margin"]),
                        "solver_status": result["solver_status"],
                        "fallback_triggered": bool(result["fallback_triggered"]),
                    }
                )
            turnover_rows.append({"date": dt, "strategy": strategy, "turnover": float(turnover)})

    weights_df = pd.DataFrame(weights_rows)
    ctr_df = pd.DataFrame(ctr_rows)
    ctb_df = pd.DataFrame(ctb_rows)
    drdb_df = pd.DataFrame(drdb_rows)
    obj_df = pd.DataFrame(obj_rows)
    turnover_df = pd.DataFrame(turnover_rows)
    perf_df = pd.DataFrame(perf_rows)
    trace_df = pd.DataFrame(trace_rows)

    # Write core outputs.
    written_files = []
    for name, df in [
        ("weights.csv", weights_df),
        ("ctr_long.csv", ctr_df),
        ("ctb_long.csv", ctb_df),
        ("dr_db_timeseries.csv", drdb_df),
        ("objective_terms.csv", obj_df),
        ("turnover_timeseries.csv", turnover_df),
        ("perf_daily.csv", perf_df),
        ("solver_trace.csv", trace_df),
    ]:
        path = panels_dir / name
        df.to_csv(path, index=False)
        written_files.append(str(path))

    summary_df = summarize_performance(perf_df, drdb_df, turnover_df)
    summary_path = summaries_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    written_files.append(str(summary_path))

    diagnostics = aggregate_diagnostics(solver_records)
    save_json(summaries_dir / "diagnostics.json", diagnostics)
    written_files.append(str(summaries_dir / "diagnostics.json"))

    # High-level analysis pack.
    best_strategy = None
    if not summary_df.empty and "sharpe" in summary_df:
        best_row = summary_df.sort_values("sharpe", ascending=False).iloc[0]
        best_strategy = best_row["strategy"]

    analysis_pack = {
        "run_id": run_dir.name,
        "best_strategy": best_strategy,
        "strategy_order": summary_df["strategy"].tolist() if not summary_df.empty else [],
        "key_findings": [
            "9A first runnable pipeline completed.",
            "Use smoke results only for pipeline validation, not for economic interpretation." if cfg["data"]["source"] == "synthetic" else "Results use local ETF price input.",
        ],
        "metric_deltas_vs_erc": {},
        "db_reduction_vs_erc": {},
        "dr_change_vs_erc": {},
        "active_rate": {r["strategy"]: r["active_rate"] for _, r in summary_df.iterrows()} if not summary_df.empty else {},
        "recommended_figures": [
            "fig_01_capital_vs_risk",
            "fig_02_dr_db_frontier",
            "fig_03_cumulative_nav",
            "fig_05_ctr_heatmap",
            "fig_06_ctb_heatmap",
            "fig_07_dr_db_timeseries",
            "fig_10_solver_convergence",
        ],
        "warnings": diagnostics.get("warning_dates", []),
    }

    if not summary_df.empty and (summary_df["strategy"] == "ERC").any():
        erc_row = summary_df.loc[summary_df["strategy"] == "ERC"].iloc[0]
        for _, row in summary_df.iterrows():
            strategy = row["strategy"]
            analysis_pack["metric_deltas_vs_erc"][strategy] = {
                "ann_return": float(row["ann_return"] - erc_row["ann_return"]),
                "ann_vol": float(row["ann_vol"] - erc_row["ann_vol"]),
                "sharpe": float(row["sharpe"] - erc_row["sharpe"]),
            }
            if strategy in ("ERC", "CTB_ONLY", "RB_CTB_BAND"):
                analysis_pack["db_reduction_vs_erc"][strategy] = float(erc_row["db_mean"] - row["db_mean"])
                analysis_pack["dr_change_vs_erc"][strategy] = float(row["dr_mean"] - erc_row["dr_mean"])

    save_json(summaries_dir / "analysis_pack.json", analysis_pack)
    written_files.append(str(summaries_dir / "analysis_pack.json"))

    final_parameters = {
        "delta": float(cfg["model"]["delta_band"]),
        "eta": float(cfg["model"]["eta_smooth"]),
        "gamma": float(cfg["model"]["gamma_l2"]),
    }

    return {
        "written_files": written_files,
        "final_parameters": final_parameters,
        "summary_df": summary_df,
        "weights_df": weights_df,
        "ctr_df": ctr_df,
        "ctb_df": ctb_df,
        "drdb_df": drdb_df,
        "perf_df": perf_df,
        "trace_df": trace_df,
    }