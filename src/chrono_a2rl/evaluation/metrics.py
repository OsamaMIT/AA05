"""Episode metric computation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from chrono_a2rl.common.types import EpisodeMetrics


def compute_metrics(rows: list[dict[str, Any]] | pd.DataFrame, termination_reason: str) -> EpisodeMetrics:
    """Compute aggregate metrics from rollout rows."""

    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    if df.empty:
        return EpisodeMetrics(termination_reason=termination_reason)

    speed = df["speed"].to_numpy(float)
    lateral = df["lateral_error"].to_numpy(float)
    heading = df["heading_error"].to_numpy(float)
    lap_completed = termination_reason == "lap_completed"
    lap_time = float(df["sim_time"].iloc[-1] - df["sim_time"].iloc[0])
    curb_samples = int(df.get("on_curb", pd.Series(dtype=bool)).astype(bool).sum())
    curb_penalty_total = float(df.get("curb_penalty_weight", pd.Series(dtype=float)).sum())
    return EpisodeMetrics(
        lap_completed=lap_completed,
        lap_time=lap_time,
        lap_time_formatted=format_lap_time(lap_time),
        mean_speed=float(np.mean(speed)),
        max_speed=float(np.max(speed)),
        lateral_error_rms=float(np.sqrt(np.mean(lateral**2))),
        max_lateral_error=float(np.max(np.abs(lateral))),
        heading_error_rms=float(np.sqrt(np.mean(heading**2))),
        off_track_count=int((~df["on_track"].astype(bool)).sum()),
        curb_sample_count=curb_samples,
        curb_usage_fraction=float(curb_samples / len(df)),
        curb_penalty_total=curb_penalty_total,
        control_saturation_count=int(df.get("control_saturated", pd.Series(dtype=bool)).sum()),
        termination_reason=termination_reason,
    )


def metrics_to_dict(metrics: EpisodeMetrics) -> dict[str, Any]:
    """Convert metrics dataclass to a plain dictionary."""

    return asdict(metrics)


def format_lap_time(seconds: float) -> str:
    """Format seconds as F1-style M:SS.mmm."""

    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    minutes, rem_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{minutes}:{secs:02d}.{millis:03d}"
