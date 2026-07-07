"""Programmatic controller evaluation entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from chrono_a2rl.evaluation.metrics import compute_metrics, metrics_to_dict


def evaluate_log(path: str | Path) -> dict[str, Any]:
    """Evaluate a saved rollout CSV."""

    df = pd.read_csv(path)
    reason = str(df["termination_reason"].iloc[-1]) if "termination_reason" in df else "unknown"
    return metrics_to_dict(compute_metrics(df, reason))
