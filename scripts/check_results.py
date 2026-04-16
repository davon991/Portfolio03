from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import latest_run_dir


REQUIRED_FILES = [
    "panels/weights.csv",
    "panels/ctr_long.csv",
    "panels/ctb_long.csv",
    "panels/dr_db_timeseries.csv",
    "panels/objective_terms.csv",
    "panels/turnover_timeseries.csv",
    "panels/perf_daily.csv",
    "summaries/summary_metrics.csv",
    "summaries/diagnostics.json",
    "summaries/analysis_pack.json",
    "summaries/run_manifest.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results/runs")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir is not None else latest_run_dir(args.results_root)
    if run_dir.name == "latest":
        run_dir = latest_run_dir(args.results_root)

    missing = [f for f in REQUIRED_FILES if not (run_dir / f).exists()]
    if missing:
        print("Missing files:")
        for m in missing:
            print(" -", m)
        raise SystemExit(1)

    summary = pd.read_csv(run_dir / "summaries" / "summary_metrics.csv")
    weights = pd.read_csv(run_dir / "panels" / "weights.csv")
    print("Result check passed.")
    print(f"run_dir = {run_dir}")
    print(f"strategies = {sorted(summary['strategy'].unique().tolist())}")
    print(f"weights rows = {len(weights)}")


if __name__ == "__main__":
    main()