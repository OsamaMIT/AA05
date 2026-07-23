"""Reset randomization for parallel RL racing environments."""

from __future__ import annotations

from typing import Any

import numpy as np

from chrono_a2rl.chrono_interface.reset_manager import initial_state_from_track
from chrono_a2rl.common.types import VehicleState
from chrono_a2rl.track.speed_profile import SpeedProfile
from chrono_a2rl.track.track_geometry import TrackGeometry


def sample_initial_state(
    *,
    track: TrackGeometry,
    simulation_config: dict[str, Any],
    rl_config: dict[str, Any],
    rng: np.random.Generator,
    speed_profile: SpeedProfile | None = None,
    vehicle_config: dict[str, Any] | None = None,
) -> VehicleState:
    """Sample an initial state for one episode.

    Randomized starts are useful for vectorized training because each parallel
    car can collect experience from a different part of the lap.
    """

    initial_speed = float(simulation_config.get("initial_speed", 3.0))
    heading_source = str(
        rl_config.get(
            "reset_heading_reference",
            rl_config.get("lateral_offset_reference", "centerline"),
        )
    )
    if not bool(rl_config.get("randomize_resets", False)):
        return initial_state_from_track(
            track,
            s=0.0,
            speed=initial_speed,
            lateral_offset=_base_lateral_offset(track, 0.0, rl_config),
            heading_source=heading_source,
        )

    s = _sample_s(track, rl_config, rng)
    speed = _sample_speed(
        initial_speed,
        rl_config,
        rng,
        s=s,
        speed_profile=speed_profile,
        vehicle_config=vehicle_config,
    )
    lateral_offset = _sample_lateral_offset(track, s, rl_config, rng)
    heading_error = float(
        rng.uniform(
            -float(rl_config.get("reset_heading_error_max", 0.0)),
            float(rl_config.get("reset_heading_error_max", 0.0)),
        )
    )
    return initial_state_from_track(
        track,
        s=s,
        speed=speed,
        lateral_offset=lateral_offset,
        heading_error=heading_error,
        heading_source=heading_source,
    )


def _sample_s(
    track: TrackGeometry,
    rl_config: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    mode = str(rl_config.get("reset_s_mode", "random")).lower()
    if mode == "fixed":
        return float(rl_config.get("reset_s", 0.0))

    fraction_min = float(rl_config.get("reset_s_fraction_min", 0.0))
    fraction_max = float(rl_config.get("reset_s_fraction_max", 1.0))
    fraction_min = max(0.0, min(1.0, fraction_min))
    fraction_max = max(fraction_min, min(1.0, fraction_max))
    return float(rng.uniform(fraction_min, fraction_max) * track.length)


def _sample_speed(
    initial_speed: float,
    rl_config: dict[str, Any],
    rng: np.random.Generator,
    *,
    s: float,
    speed_profile: SpeedProfile | None = None,
    vehicle_config: dict[str, Any] | None = None,
) -> float:
    mode = str(rl_config.get("reset_speed_mode", "range")).lower()
    if mode == "profile" and speed_profile is not None:
        scale_min = float(rl_config.get("reset_speed_scale_min", 1.0))
        scale_max = float(rl_config.get("reset_speed_scale_max", scale_min))
        if scale_max < scale_min:
            scale_max = scale_min
        profile_speed = speed_profile.speed_at(s)
        vehicle_max = float((vehicle_config or {}).get("max_speed", speed_profile.max_speed))
        return float(np.clip(profile_speed * rng.uniform(scale_min, scale_max), 0.0, vehicle_max))

    speed_min = float(rl_config.get("reset_speed_min", initial_speed))
    speed_max = float(rl_config.get("reset_speed_max", initial_speed))
    if speed_max < speed_min:
        speed_max = speed_min
    return float(rng.uniform(speed_min, speed_max))


def _sample_lateral_offset(
    track: TrackGeometry,
    s: float,
    rl_config: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    base_offset = _base_lateral_offset(track, s, rl_config)
    max_abs_offset = float(rl_config.get("reset_lateral_offset_max", 0.0))
    if max_abs_offset <= 0.0:
        return base_offset
    margin = float(rl_config.get("reset_lateral_margin", 1.0))
    sample = track.interpolate(s)
    left_limit = max(0.0, sample.width_left - margin)
    right_limit = max(0.0, sample.width_right - margin)
    low = max(-max_abs_offset, -right_limit - base_offset)
    high = min(max_abs_offset, left_limit - base_offset)
    if high < low:
        return base_offset
    return float(base_offset + rng.uniform(low, high))


def _base_lateral_offset(track: TrackGeometry, s: float, rl_config: dict[str, Any]) -> float:
    reference = str(
        rl_config.get("reset_lateral_reference", rl_config.get("lateral_offset_reference", "centerline"))
    ).lower()
    if reference == "raceline":
        return track.raceline_lateral_offset_at(s)
    return 0.0
