from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import ensure_dir


def _save_fig(fig, path_png: Path, path_pdf: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def make_all_figures(cfg: dict[str, Any], run_dir: str | Path) -> list[str]:
    run_dir = Path(run_dir)
    panels_dir = run_dir / "panels"
    summaries_dir = run_dir / "summaries"
    figures_dir = ensure_dir(run_dir / "figures")
    written = []

    weights = pd.read_csv(panels_dir / "weights.csv")
    ctr = pd.read_csv(panels_dir / "ctr_long.csv") if (panels_dir / "ctr_long.csv").exists() else pd.DataFrame()
    ctb = pd.read_csv(panels_dir / "ctb_long.csv") if (panels_dir / "ctb_long.csv").exists() else pd.DataFrame()
    drdb = pd.read_csv(panels_dir / "dr_db_timeseries.csv") if (panels_dir / "dr_db_timeseries.csv").exists() else pd.DataFrame()
    perf = pd.read_csv(panels_dir / "perf_daily.csv") if (panels_dir / "perf_daily.csv").exists() else pd.DataFrame()
    trace = pd.read_csv(panels_dir / "solver_trace.csv") if (panels_dir / "solver_trace.csv").exists() else pd.DataFrame()
    summary = pd.read_csv(summaries_dir / "summary_metrics.csv") if (summaries_dir / "summary_metrics.csv").exists() else pd.DataFrame()

    # fig_01 capital vs risk
    if not weights.empty and not ctr.empty:
        latest_date = sorted(weights["date"].unique())[-1]
        w = weights[(weights["date"] == latest_date) & (weights["strategy"] == "RB_CTB_BAND")][["asset", "weight"]].copy()
        r = ctr[(ctr["date"] == latest_date) & (ctr["strategy"] == "RB_CTB_BAND")][["asset", "ctr"]].copy()
        fig1_df = w.merge(r, on="asset", how="left").fillna(0.0)
        fig1_df.to_csv(figures_dir / "fig_01_capital_vs_risk.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(fig1_df["asset"], fig1_df["weight"], label="Capital weight")
        ax.bar(fig1_df["asset"], fig1_df["ctr"], alpha=0.6, label="Risk contribution")
        ax.set_title("Capital vs Risk Allocation (RB_CTB_BAND, latest rebalance)")
        ax.legend()
        _save_fig(fig, figures_dir / "fig_01_capital_vs_risk.png", figures_dir / "fig_01_capital_vs_risk.pdf")
        written += [str(figures_dir / "fig_01_capital_vs_risk.png"), str(figures_dir / "fig_01_capital_vs_risk.pdf")]

    # fig_02 dr_db frontier
    if not summary.empty:
        fig2_df = summary[["strategy", "dr_mean", "db_mean"]].copy()
        fig2_df.to_csv(figures_dir / "fig_02_dr_db_frontier.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(fig2_df["db_mean"], fig2_df["dr_mean"])
        for _, row in fig2_df.iterrows():
            ax.annotate(row["strategy"], (row["db_mean"], row["dr_mean"]))
        ax.set_xlabel("DB mean")
        ax.set_ylabel("DR mean")
        ax.set_title("DR-DB Frontier")
        _save_fig(fig, figures_dir / "fig_02_dr_db_frontier.png", figures_dir / "fig_02_dr_db_frontier.pdf")
        written += [str(figures_dir / "fig_02_dr_db_frontier.png"), str(figures_dir / "fig_02_dr_db_frontier.pdf")]

    # fig_03 cumulative nav
    if not perf.empty:
        nav_df = perf[["date", "strategy", "nav"]].copy()
        nav_df.to_csv(figures_dir / "fig_03_cumulative_nav.csv", index=False)
        pivot = nav_df.pivot(index="date", columns="strategy", values="nav")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        pivot.plot(ax=ax)
        ax.set_title("Cumulative NAV")
        ax.set_ylabel("NAV")
        _save_fig(fig, figures_dir / "fig_03_cumulative_nav.png", figures_dir / "fig_03_cumulative_nav.pdf")
        written += [str(figures_dir / "fig_03_cumulative_nav.png"), str(figures_dir / "fig_03_cumulative_nav.pdf")]

    # fig_05 ctr heatmap
    if not ctr.empty:
        latest = ctr.pivot_table(index="asset", columns="date", values="ctr", aggfunc="mean")
        latest.to_csv(figures_dir / "fig_05_ctr_heatmap.csv")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(latest.to_numpy(), aspect="auto")
        ax.set_yticks(range(len(latest.index)))
        ax.set_yticklabels(latest.index)
        ax.set_xticks(range(len(latest.columns)))
        ax.set_xticklabels([str(c)[:10] for c in latest.columns], rotation=90)
        ax.set_title("CtR Heatmap")
        fig.colorbar(im, ax=ax)
        _save_fig(fig, figures_dir / "fig_05_ctr_heatmap.png", figures_dir / "fig_05_ctr_heatmap.pdf")
        written += [str(figures_dir / "fig_05_ctr_heatmap.png"), str(figures_dir / "fig_05_ctr_heatmap.pdf")]

    # fig_06 ctb heatmap
    if not ctb.empty:
        latest = ctb.pivot_table(index="asset", columns="date", values="ctb", aggfunc="mean")
        latest.to_csv(figures_dir / "fig_06_ctb_heatmap.csv")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(latest.to_numpy(), aspect="auto")
        ax.set_yticks(range(len(latest.index)))
        ax.set_yticklabels(latest.index)
        ax.set_xticks(range(len(latest.columns)))
        ax.set_xticklabels([str(c)[:10] for c in latest.columns], rotation=90)
        ax.set_title("CtB Heatmap")
        fig.colorbar(im, ax=ax)
        _save_fig(fig, figures_dir / "fig_06_ctb_heatmap.png", figures_dir / "fig_06_ctb_heatmap.pdf")
        written += [str(figures_dir / "fig_06_ctb_heatmap.png"), str(figures_dir / "fig_06_ctb_heatmap.pdf")]

    # fig_07 dr db timeseries
    if not drdb.empty:
        drdb.to_csv(figures_dir / "fig_07_dr_db_timeseries.csv", index=False)
        pivot_dr = drdb.pivot(index="date", columns="strategy", values="dr")
        pivot_db = drdb.pivot(index="date", columns="strategy", values="db")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        pivot_dr.plot(ax=axes[0])
        axes[0].set_title("DR Timeseries")
        pivot_db.plot(ax=axes[1])
        axes[1].set_title("DB Timeseries")
        _save_fig(fig, figures_dir / "fig_07_dr_db_timeseries.png", figures_dir / "fig_07_dr_db_timeseries.pdf")
        written += [str(figures_dir / "fig_07_dr_db_timeseries.png"), str(figures_dir / "fig_07_dr_db_timeseries.pdf")]

    # fig_10 solver convergence
    if not trace.empty:
        trace.to_csv(figures_dir / "fig_10_solver_convergence.csv", index=False)
        tmp = trace.groupby(["phase", "iter"], as_index=False)[["objective_value", "grad_norm"]].mean()
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
        for phase, g in tmp.groupby("phase"):
            axes[0].plot(g["iter"], g["objective_value"], label=phase)
            axes[1].plot(g["iter"], g["grad_norm"], label=phase)
        axes[0].set_title("Solver Convergence: objective")
        axes[1].set_title("Solver Convergence: grad norm")
        axes[0].legend()
        axes[1].legend()
        _save_fig(fig, figures_dir / "fig_10_solver_convergence.png", figures_dir / "fig_10_solver_convergence.pdf")
        written += [str(figures_dir / "fig_10_solver_convergence.png"), str(figures_dir / "fig_10_solver_convergence.pdf")]

    return written