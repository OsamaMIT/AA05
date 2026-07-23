"""Curvature-based target speed profile generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chrono_a2rl.common.config import resolve_path
from chrono_a2rl.common.math_utils import circular_moving_average
from chrono_a2rl.track.track_geometry import TrackGeometry


@dataclass(slots=True)
class SpeedProfile:
    """Closed-loop target speed profile."""

    s: np.ndarray
    speed: np.ndarray
    track_length: float
    min_speed: float
    max_speed: float

    def speed_at(self, s_value: float) -> float:
        """Interpolate target speed at arc length."""

        s_mod = float(s_value % self.track_length)
        s_ext = np.concatenate([self.s, [self.track_length]])
        v_ext = np.concatenate([self.speed, [self.speed[0]]])
        return float(np.interp(s_mod, s_ext, v_ext))

    def target_speed_at(self, s_value: float, speed_scale: float = 1.0) -> float:
        """Return a scaled target speed clipped to the configured vehicle envelope."""

        scaled = self.speed_at(s_value) * float(speed_scale)
        return float(np.clip(scaled, 0.0, self.max_speed))


def generate_speed_profile(track: TrackGeometry, config: dict[str, Any]) -> SpeedProfile:
    """Generate a curvature-limited closed-loop target speed profile.

    Optional longitudinal acceleration and braking limits smooth the raw
    curvature cap so high-speed profiles start slowing down before tight bends.
    The limits are still lightweight research scaffolding, not a full
    time-optimal velocity planner.
    """

    profile_path = config.get("profile_path")
    if profile_path:
        return load_speed_profile(track, profile_path, config)

    max_speed = float(config.get("max_speed", 15.0))
    min_speed = float(config.get("min_speed", 5.0))
    max_lateral_accel = float(config.get("max_lateral_accel", 3.5))
    shallow_max_lateral_accel = float(
        config.get("shallow_curve_max_lateral_accel", max_lateral_accel)
    )
    shallow_curvature_full = float(
        config.get("shallow_curve_curvature_full", 0.0)
    )
    shallow_curvature_end = float(
        config.get("shallow_curve_curvature_end", shallow_curvature_full)
    )
    max_accel = float(config.get("max_accel", 0.0))
    max_decel = float(config.get("max_decel", 0.0))
    smoothing_window = int(config.get("smoothing_window", 1))
    epsilon = float(config.get("curvature_epsilon", 1.0e-4))

    curvature_source = str(config.get("curvature_source", "centerline")).lower()
    if curvature_source not in {"centerline", "raceline"}:
        raise ValueError(
            "speed_profile.curvature_source must be 'centerline' or 'raceline'"
        )
    if curvature_source == "centerline":
        curvature = np.abs(track.curvature)
    else:
        curvature = np.abs(
            np.asarray(
                [track.raceline_curvature_at(float(s)) for s in track.s_nodes],
                dtype=float,
            )
        )
    lateral_accel_limit = _lateral_accel_envelope(
        curvature,
        base_limit=max_lateral_accel,
        shallow_limit=shallow_max_lateral_accel,
        shallow_curvature_full=shallow_curvature_full,
        shallow_curvature_end=shallow_curvature_end,
    )
    raw = np.sqrt(lateral_accel_limit / np.maximum(curvature, epsilon))
    speed = np.clip(raw, min_speed, max_speed)
    speed = circular_moving_average(speed, smoothing_window)
    speed = np.clip(speed, min_speed, max_speed)
    speed = _apply_longitudinal_limits(speed, track.s_nodes, track.length, max_accel, max_decel)
    speed = np.clip(speed, min_speed, max_speed)
    return SpeedProfile(
        s=track.s_nodes.copy(),
        speed=speed,
        track_length=track.length,
        min_speed=min_speed,
        max_speed=max_speed,
    )


def _lateral_accel_envelope(
    curvature: np.ndarray,
    *,
    base_limit: float,
    shallow_limit: float,
    shallow_curvature_full: float,
    shallow_curvature_end: float,
) -> np.ndarray:
    """Blend a higher lateral limit into shallow raceline bends."""

    if (
        shallow_limit <= base_limit
        or shallow_curvature_full < 0.0
        or shallow_curvature_end <= shallow_curvature_full
    ):
        return np.full_like(curvature, base_limit, dtype=float)
    blend = np.clip(
        (curvature - shallow_curvature_full)
        / (shallow_curvature_end - shallow_curvature_full),
        0.0,
        1.0,
    )
    smooth_blend = blend * blend * (3.0 - 2.0 * blend)
    return shallow_limit + (base_limit - shallow_limit) * smooth_blend


def load_speed_profile(
    track: TrackGeometry,
    path: str | Path,
    config: dict[str, Any],
) -> SpeedProfile:
    """Load an optimized closed-loop profile stored in human-readable km/h."""

    profile_path = resolve_path(path)
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Optimized speed profile not found: {profile_path}. "
            "Run `aa optimize` or remove speed_profile.profile_path."
        )
    data = pd.read_csv(profile_path)
    if "s_m" not in data:
        raise ValueError(f"Optimized profile must contain s_m: {profile_path}")
    if "speed_kmh" in data:
        source_speed = data["speed_kmh"].to_numpy(float) / 3.6
    elif "speed_mps" in data:
        source_speed = data["speed_mps"].to_numpy(float)
    else:
        raise ValueError(
            f"Optimized profile must contain speed_kmh or speed_mps: {profile_path}"
        )
    source_s = data["s_m"].to_numpy(float)
    if source_s.size < 2 or source_s.size != source_speed.size:
        raise ValueError(f"Optimized profile contains insufficient samples: {profile_path}")
    if not np.all(np.isfinite(source_s)) or not np.all(np.isfinite(source_speed)):
        raise ValueError(f"Optimized profile contains non-finite values: {profile_path}")

    order = np.argsort(source_s)
    source_s = np.mod(source_s[order], track.length)
    source_speed = source_speed[order]
    unique_s, unique_indices = np.unique(source_s, return_index=True)
    source_speed = source_speed[unique_indices]
    s_ext = np.concatenate([unique_s, [unique_s[0] + track.length]])
    speed_ext = np.concatenate([source_speed, [source_speed[0]]])
    query_s = track.s_nodes.copy()
    query_s[query_s < unique_s[0]] += track.length

    min_speed = float(config.get("min_speed", 0.0))
    max_speed = float(config.get("max_speed", np.max(source_speed)))
    speed = np.interp(query_s, s_ext, speed_ext)
    speed = np.clip(speed, min_speed, max_speed)
    return SpeedProfile(
        s=track.s_nodes.copy(),
        speed=speed,
        track_length=track.length,
        min_speed=min_speed,
        max_speed=max_speed,
    )


def profile_from_speed_samples(
    track: TrackGeometry,
    speed: np.ndarray,
    config: dict[str, Any],
    *,
    apply_longitudinal_limits: bool = True,
) -> SpeedProfile:
    """Build a valid profile from one speed value per track geometry node."""

    values = np.asarray(speed, dtype=float)
    if values.shape != track.s_nodes.shape:
        raise ValueError(
            f"speed samples must have shape {track.s_nodes.shape}, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("speed samples must be finite")
    min_speed = float(config.get("min_speed", 0.0))
    max_speed = float(config.get("max_speed", np.max(values)))
    values = np.clip(values, min_speed, max_speed)
    if apply_longitudinal_limits:
        values = _apply_longitudinal_limits(
            values,
            track.s_nodes,
            track.length,
            float(config.get("max_accel", 0.0)),
            float(config.get("max_decel", 0.0)),
        )
    return SpeedProfile(
        s=track.s_nodes.copy(),
        speed=np.clip(values, min_speed, max_speed),
        track_length=track.length,
        min_speed=min_speed,
        max_speed=max_speed,
    )


def _apply_longitudinal_limits(
    speed: np.ndarray,
    s_nodes: np.ndarray,
    track_length: float,
    max_accel: float,
    max_decel: float,
) -> np.ndarray:
    """Apply simple closed-loop acceleration and deceleration feasibility limits."""

    limited = speed.astype(float, copy=True)
    if limited.size < 2:
        return limited

    ds = np.diff(np.concatenate([s_nodes, [track_length]]))
    ds = np.maximum(ds, 1.0e-6)

    for _ in range(3):
        if max_accel > 0.0:
            for i in range(1, limited.size):
                limited[i] = min(
                    limited[i],
                    np.sqrt(max(0.0, limited[i - 1] ** 2 + 2.0 * max_accel * ds[i - 1])),
                )
            limited[0] = min(
                limited[0],
                np.sqrt(max(0.0, limited[-1] ** 2 + 2.0 * max_accel * ds[-1])),
            )

        if max_decel > 0.0:
            for i in range(limited.size - 2, -1, -1):
                limited[i] = min(
                    limited[i],
                    np.sqrt(max(0.0, limited[i + 1] ** 2 + 2.0 * max_decel * ds[i])),
                )
            limited[-1] = min(
                limited[-1],
                np.sqrt(max(0.0, limited[0] ** 2 + 2.0 * max_decel * ds[-1])),
            )

    return limited
