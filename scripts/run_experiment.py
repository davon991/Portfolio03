from __future__ import annotations

import argparse
from pathlib import Path

from src.backtest import run_full_pipeline
from src.data_prep import prepare_returns
from src.reporting import make_all_figures
from src.utils import compute_run_id, ensure_dir, load_yaml, save_json, set_latest_pointer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    results_root = ensure_dir(cfg["run"]["results_dir"])
    run_id = compute_run_id(cfg)
    run_dir = ensure_dir(Path(results_root) / run_id)

    # Save config snapshot.
    config_snapshot_dir = ensure_dir(run_dir / "config_snapshot")
    src_cfg_path = Path(args.config)
    (config_snapshot_dir / src_cfg_path.name).write_text(src_cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

    data_artifacts = prepare_returns(cfg)
    outputs = run_full_pipeline(cfg, run_dir, data_artifacts)
    figure_files = make_all_figures(cfg, run_dir) if cfg.get("reporting", {}).get("make_figures", True) else []

    run_manifest = {
        "run_id": run_id,
        "config_file": args.config,
        "data_source": cfg["data"]["source"],
        "strategies": cfg["experiment"]["strategies"],
        "output_files": outputs["written_files"] + figure_files,
    }
    save_json(run_dir / "summaries" / "run_manifest.json", run_manifest)

    set_latest_pointer(results_root, run_id, cfg["run"].get("latest_symlink_name", "latest"))
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()