"""Episode metric computation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from chrono_a2rl.common.types import EpisodeMetrics

MPS_TO_KMH = 3.6


def compute_metrics(rows: list[dict[str, Any]] | pd.DataFrame, termination_reason: str) -> EpisodeMetrics:
    """Compute aggregate metrics from rollout rows."""

    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    if df.empty:
        return EpisodeMetrics(termination_reason=termination_reason)

    speed = _speed_mps_from_rows(df)
    lateral = _numeric_column(df, ("raceline_error", "lateral_error"))
    heading = df["heading_error"].to_numpy(float)
    lap_completed = termination_reason == "lap_completed"
    sim_time_seconds = _sim_time_seconds_from_rows(df)
    lap_time = (
        float(df["episode_time_seconds"].iloc[-1])
        if "episode_time_seconds" in df
        else float(sim_time_seconds[-1] - sim_time_seconds[0])
    )
    curb_samples = int(df.get("on_curb", pd.Series(dtype=bool)).astype(bool).sum())
    curb_penalty_total = float(df.get("curb_penalty_weight", pd.Series(dtype=float)).sum())
    longitudinal_mode = (
        "action_mode" in df
        and df["action_mode"]
        .astype(str)
        .isin({"longitudinal_pedal", "profile_pedal_residual"})
        .any()
    )
    speed_scale = (
        np.asarray([], dtype=float)
        if longitudinal_mode
        else _numeric_column(df, ("effective_speed_scale", "speed_scale", "action_0"))
    )
    target_speed_kmh = _target_speed_kmh_from_rows(df)
    profile_speed_error_kmh = (
        speed * MPS_TO_KMH - target_speed_kmh
        if target_speed_kmh.size == speed.size
        else np.asarray([], dtype=float)
    )
    longitudinal_action = _numeric_column(df, ("longitudinal_action",))
    throttle = _numeric_column(df, ("applied_throttle", "throttle"))
    brake = _numeric_column(df, ("applied_brake", "brake"))
    completed_corners = df[
        df.get("corner_completed", pd.Series(False, index=df.index)).astype(bool)
    ]
    corner_scores = _numeric_column(completed_corners, ("corner_score",))
    apex_speeds = _numeric_column(completed_corners, ("apex_speed_kmh",))
    exit_speeds = _numeric_column(completed_corners, ("exit_speed_kmh",))
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
        mean_speed_scale=float(np.mean(speed_scale)) if speed_scale.size else 0.0,
        min_speed_scale=float(np.min(speed_scale)) if speed_scale.size else 0.0,
        max_speed_scale=float(np.max(speed_scale)) if speed_scale.size else 0.0,
        mean_target_speed_kmh=(
            float(np.mean(target_speed_kmh)) if target_speed_kmh.size else 0.0
        ),
        max_target_speed_kmh=(
            float(np.max(target_speed_kmh)) if target_speed_kmh.size else 0.0
        ),
        profile_speed_error_rmse_kmh=(
            float(np.sqrt(np.mean(profile_speed_error_kmh**2)))
            if profile_speed_error_kmh.size
            else 0.0
        ),
        profile_speed_error_mae_kmh=(
            float(np.mean(np.abs(profile_speed_error_kmh)))
            if profile_speed_error_kmh.size
            else 0.0
        ),
        mean_longitudinal_action=(
            float(np.mean(longitudinal_action)) if longitudinal_action.size else 0.0
        ),
        min_longitudinal_action=(
            float(np.min(longitudinal_action)) if longitudinal_action.size else 0.0
        ),
        max_longitudinal_action=(
            float(np.max(longitudinal_action)) if longitudinal_action.size else 0.0
        ),
        mean_throttle=float(np.mean(throttle)) if throttle.size else 0.0,
        mean_brake=float(np.mean(brake)) if brake.size else 0.0,
        braking_fraction=(
            float(np.mean(brake > 0.01)) if brake.size else 0.0
        ),
        max_validated_progress_m=float(
            df.get("validated_progress_m", pd.Series([0.0])).max()
        ),
        frontier_progress_m=float(
            df.get("frontier_progress_m", pd.Series([0.0])).iloc[-1]
        ),
        frontier_advancement_m=float(
            df.get("frontier_advancement_m", pd.Series([0.0])).sum()
        ),
        frontier_cleared=bool(
            df.get("frontier_cleared", pd.Series(False, index=df.index)).astype(bool).any()
        ),
        training_role=str(
            df.get("training_role", pd.Series(["evaluation"])).iloc[0]
        ),
        corner_completion_count=int(
            df.get("corner_completed", pd.Series(False, index=df.index)).astype(bool).sum()
        ),
        mean_corner_score=float(np.mean(corner_scores)) if corner_scores.size else 0.0,
        max_corner_score=float(np.max(corner_scores)) if corner_scores.size else 0.0,
        mean_apex_speed_kmh=float(np.mean(apex_speeds)) if apex_speeds.size else 0.0,
        mean_exit_speed_kmh=float(np.mean(exit_speeds)) if exit_speeds.size else 0.0,
        kinetic_crash_penalty=float(
            df.get("kinetic_crash_penalty", pd.Series([0.0])).sum()
        ),
        termination_reason=termination_reason,
    )


def metrics_to_dict(metrics: EpisodeMetrics) -> dict[str, Any]:
    """Convert metrics dataclass to a human-facing dictionary.

    `lap_time` is formatted as M:SS.mmm for console/YAML output. The raw
    numeric value is preserved as `lap_time_seconds`. Speeds are reported only
    in km/h.
    """

    data = asdict(metrics)
    lap_time_seconds = data.pop("lap_time")
    lap_time_formatted = data.pop("lap_time_formatted")
    mean_speed_kmh = float(data.pop("mean_speed")) * MPS_TO_KMH
    max_speed_kmh = float(data.pop("max_speed")) * MPS_TO_KMH
    return {
        "lap_completed": data.pop("lap_completed"),
        "lap_time": lap_time_formatted,
        "lap_time_seconds": lap_time_seconds,
        "mean_speed_kmh": mean_speed_kmh,
        "max_speed_kmh": max_speed_kmh,
        **data,
    }


def format_lap_time(seconds: float) -> str:
    """Format seconds as F1-style M:SS.mmm."""

    total_ms = int(round(max(0.0, float(seconds)) * 1000.0))
    minutes, rem_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{minutes}:{secs:02d}.{millis:03d}"


def _speed_mps_from_rows(df: pd.DataFrame) -> np.ndarray:
    """Read speed from old m/s rows or newer km/h report rows."""

    if "speed" in df:
        return df["speed"].to_numpy(float)
    if "speed_mps" in df:
        return df["speed_mps"].to_numpy(float)
    if "speed_kmh" in df:
        return df["speed_kmh"].to_numpy(float) / MPS_TO_KMH
    raise KeyError("rollout rows must contain speed_kmh, speed_mps, or speed")


def _sim_time_seconds_from_rows(df: pd.DataFrame) -> np.ndarray:
    """Read numeric simulation time from new or legacy rollout logs."""

    if "sim_time_seconds" in df:
        return df["sim_time_seconds"].to_numpy(float)
    if "sim_time" not in df:
        raise KeyError("rollout rows must contain sim_time or sim_time_seconds")

    values = df["sim_time"]
    try:
        return values.to_numpy(float)
    except (TypeError, ValueError):
        return np.asarray([_parse_f1_time(value) for value in values], dtype=float)


def _parse_f1_time(value: Any) -> float:
    """Parse M:SS.mmm while retaining compatibility with numeric strings."""

    text = str(value).strip()
    if ":" not in text:
        return float(text)
    minutes, seconds = text.split(":", maxsplit=1)
    return float(minutes) * 60.0 + float(seconds)


def _target_speed_kmh_from_rows(df: pd.DataFrame) -> np.ndarray:
    """Read target speed from rollout rows in km/h."""

    if "target_speed_kmh" in df:
        return df["target_speed_kmh"].to_numpy(float)
    if "target_speed" in df:
        return df["target_speed"].to_numpy(float) * MPS_TO_KMH
    return np.asarray([], dtype=float)


def _numeric_column(df: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    """Return the first available numeric column from a list of candidates."""

    for name in names:
        if name in df:
            return df[name].to_numpy(float)
    return np.asarray([], dtype=float)
