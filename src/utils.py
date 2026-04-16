from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_run_id(cfg: dict[str, Any]) -> str:
    now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    short_hash = hashlib.sha256(payload).hexdigest()[:8]
    return f"{now}-{short_hash}"


def set_latest_pointer(results_root: str | Path, run_id: str, latest_name: str = "latest") -> None:
    root = Path(results_root)
    latest_path = root / latest_name
    target = root / run_id
    if latest_path.exists() or latest_path.is_symlink():
        if latest_path.is_symlink() or latest_path.is_file():
            latest_path.unlink()
        else:
            # keep behavior simple on Windows
            pass
    try:
        latest_path.symlink_to(target.resolve(), target_is_directory=True)
    except Exception:
        # Windows often blocks symlinks without admin; use a plain text pointer.
        latest_txt = root / f"{latest_name}.txt"
        latest_txt.write_text(str(target.resolve()), encoding="utf-8")


def safe_to_parquet(df, path: str | Path) -> str:
    path = Path(path)
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False)
        return str(path)
    except Exception:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return str(csv_path)


def latest_run_dir(results_root: str | Path) -> Path:
    root = Path(results_root)
    latest_txt = root / "latest.txt"
    if latest_txt.exists():
        return Path(latest_txt.read_text(encoding="utf-8").strip())
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError("No run directories found under results_root.")
    return max(runs, key=lambda p: p.stat().st_mtime)