"""Run-directory and latest-model resolution for planner training."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


LATEST_RUN_FILE = "latest_run.yaml"


def create_run_directory(model_root: str | Path, *, seed: int) -> tuple[str, Path]:
    """Create a unique timestamped run directory and mark it latest."""

    root = Path(model_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_id = f"{stamp}_seed{seed}"
    run_id = base_id
    suffix = 1
    while (root / run_id).exists():
        suffix += 1
        run_id = f"{base_id}_{suffix}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    mark_latest_run(root, run_dir)
    return run_id, run_dir


def mark_latest_run(model_root: str | Path, run_dir: str | Path) -> Path:
    """Atomically update the lightweight latest-run pointer."""

    root = Path(model_root)
    run_path = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        relative = run_path.relative_to(root.resolve())
        stored_path = str(relative)
    except ValueError:
        stored_path = str(run_path)
    pointer = root / LATEST_RUN_FILE
    temporary = pointer.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(
            {
                "run_dir": stored_path,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(pointer)
    return pointer


def resolve_latest_run(model_root: str | Path) -> Path:
    """Resolve the latest planner run from its pointer or directory timestamps."""

    root = Path(model_root)
    pointer = root / LATEST_RUN_FILE
    if pointer.exists():
        data = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
        raw_path = Path(str(data.get("run_dir", "")))
        candidate = raw_path if raw_path.is_absolute() else root / raw_path
        if candidate.is_dir():
            return candidate
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No planner runs found in {root}. Start one with `aa train`.")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def resolve_latest_model(model_root: str | Path) -> Path:
    """Resolve final_model.zip or the newest checkpoint in the latest run."""

    run_dir = resolve_latest_run(model_root)
    final_model = run_dir / "final_model.zip"
    if final_model.exists():
        return final_model
    checkpoints = list(run_dir.glob("*_steps.zip"))
    if not checkpoints:
        raise FileNotFoundError(f"No models found in latest run: {run_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)


def resolve_model_spec(model: str | Path, model_root: str | Path) -> Path:
    """Resolve `latest` or validate an explicit model archive path."""

    if str(model).strip().lower() == "latest":
        return resolve_latest_model(model_root)
    path = Path(model).expanduser()
    if not path.suffix and path.with_suffix(".zip").exists():
        path = path.with_suffix(".zip")
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return path


def write_run_manifest(
    run_dir: str | Path,
    *,
    run_id: str,
    config: dict[str, Any],
) -> Path:
    """Save the resolved training configuration for reproducibility."""

    output = Path(run_dir) / "training_config.yaml"
    output.write_text(
        yaml.safe_dump({"run_id": run_id, "config": config}, sort_keys=False),
        encoding="utf-8",
    )
    return output
