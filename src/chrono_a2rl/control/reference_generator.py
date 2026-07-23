"""Controller reference generation."""

from __future__ import annotations

import math

from chrono_a2rl.common.types import ControllerReference, TrackState
from chrono_a2rl.track.speed_profile import SpeedProfile
from chrono_a2rl.track.track_geometry import TrackGeometry


def make_reference(
    track: TrackGeometry,
    speed_profile: SpeedProfile,
    track_state: TrackState,
    *,
    lookahead_time: float = 0.25,
    speed_scale: float = 1.0,
    lateral_offset: float = 0.0,
    lookahead_lateral_offset: float | None = None,
    path_source: str = "centerline",
    target_speed_override: float | None = None,
    horizon_steps: int = 0,
    control_dt: float = 0.02,
) -> ControllerReference:
    """Build a local reference with distinct current and preview offsets.

    `lateral_offset` is expressed in the vehicle's current Frenet cross-section
    and is consumed by the lateral controller. `lookahead_lateral_offset`
    places the preview point on the future path without corrupting the current
    lateral-error calculation.
    """

    target_speed = (
        speed_profile.target_speed_at(track_state.s, speed_scale)
        if target_speed_override is None
        else max(0.0, min(speed_profile.max_speed, float(target_speed_override)))
    )
    lookahead_s = track_state.s + max(target_speed, 0.0) * lookahead_time
    sample = track.interpolate(lookahead_s)
    current_sample = track.interpolate(track_state.s)
    lateral_offset = max(
        -current_sample.width_right,
        min(current_sample.width_left, lateral_offset),
    )
    preview_offset = (
        lateral_offset
        if lookahead_lateral_offset is None
        else float(lookahead_lateral_offset)
    )
    preview_offset = max(
        -sample.width_right,
        min(sample.width_left, preview_offset),
    )
    use_raceline = path_source.lower() == "raceline" and track.raceline is not None
    if use_raceline and lookahead_lateral_offset is not None:
        target_x, target_y = track.raceline_xy_at(sample.s)
        target_yaw = track.raceline_heading_at(sample.s)
        target_curvature = track.raceline_curvature_at(sample.s)
        current_target_yaw = track.raceline_heading_at(track_state.s)
        current_target_curvature = track.raceline_curvature_at(track_state.s)
    else:
        normal_x = -math.sin(sample.heading)
        normal_y = math.cos(sample.heading)
        target_x = sample.x + preview_offset * normal_x
        target_y = sample.y + preview_offset * normal_y
        target_yaw = sample.heading
        target_curvature = sample.curvature
        current_sample = track.interpolate(track_state.s)
        current_target_yaw = current_sample.heading
        current_target_curvature = current_sample.curvature
    curvature_preview = tuple(
        float(
            track.curvature_at(
                track_state.s + max(target_speed, 0.5) * control_dt * (step + 1),
                source="raceline" if use_raceline else "centerline",
            )
        )
        for step in range(max(0, int(horizon_steps)))
    )
    return ControllerReference(
        target_speed=target_speed,
        target_s=sample.s,
        target_x=float(target_x),
        target_y=float(target_y),
        target_yaw=float(target_yaw),
        target_curvature=float(target_curvature),
        target_lateral_offset=lateral_offset,
        current_target_yaw=float(current_target_yaw),
        current_target_curvature=float(current_target_curvature),
        curvature_preview=curvature_preview,
    )


def offset_from_fraction(
    track: TrackGeometry,
    s: float,
    offset_fraction: float,
    *,
    margin: float = 1.0,
) -> float:
    """Convert normalized lateral offset fraction to meters within track limits."""

    sample = track.interpolate(s)
    left_limit = max(0.0, sample.width_left - margin)
    right_limit = max(0.0, sample.width_right - margin)
    fraction = max(-1.0, min(1.0, offset_fraction))
    if fraction >= 0.0:
        return fraction * left_limit
    return fraction * right_limit
