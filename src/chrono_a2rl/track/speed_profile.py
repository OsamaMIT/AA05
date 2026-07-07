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

    def speed_at(self, s_value: float) -> float:
        """Interpolate target speed at arc length."""

        s_mod = float(s_value % self.track_length)
        s_ext = np.concatenate([self.s, [self.track_length]])
        v_ext = np.concatenate([self.speed, [self.speed[0]]])
        return float(np.interp(s_mod, s_ext, v_ext))


def generate_speed_profile(track: TrackGeometry, config: dict[str, Any]) -> SpeedProfile:
    """Generate a conservative curvature-limited speed profile."""

    max_speed = float(config.get("max_speed", 15.0))
    min_speed = float(config.get("min_speed", 5.0))
    max_lateral_accel = float(config.get("max_lateral_accel", 3.5))
    smoothing_window = int(config.get("smoothing_window", 1))
    epsilon = float(config.get("curvature_epsilon", 1.0e-4))

    curvature = np.abs(track.curvature)
    raw = np.sqrt(max_lateral_accel / np.maximum(curvature, epsilon))
    speed = np.clip(raw, min_speed, max_speed)
    speed = circular_moving_average(speed, smoothing_window)
    speed = np.clip(speed, min_speed, max_speed)
    return SpeedProfile(s=track.s_nodes.copy(), speed=speed, track_length=track.length)
