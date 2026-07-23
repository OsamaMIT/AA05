"""Curvature-based target speed profile generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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

    max_speed = float(config.get("max_speed", 15.0))
    min_speed = float(config.get("min_speed", 5.0))
    max_lateral_accel = float(config.get("max_lateral_accel", 3.5))
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
    raw = np.sqrt(max_lateral_accel / np.maximum(curvature, epsilon))
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
