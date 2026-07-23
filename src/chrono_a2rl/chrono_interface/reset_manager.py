"""Initial-state helpers for closed-loop simulations."""

from __future__ import annotations

import math

from chrono_a2rl.common.math_utils import wrap_angle
from chrono_a2rl.common.types import VehicleState
from chrono_a2rl.track.track_geometry import TrackGeometry


def initial_state_from_track(
    track: TrackGeometry,
    *,
    s: float = 0.0,
    speed: float = 0.0,
    lateral_offset: float = 0.0,
    heading_error: float = 0.0,
    heading_source: str = "centerline",
) -> VehicleState:
    """Create an initial VehicleState aligned with a track-relative pose."""

    sample = track.interpolate(s)
    reference_heading = (
        track.raceline_heading_at(s)
        if heading_source.lower() == "raceline" and track.raceline is not None
        else sample.heading
    )
    yaw = wrap_angle(reference_heading + heading_error)
    normal_x = -math.sin(sample.heading)
    normal_y = math.cos(sample.heading)
    x = sample.x + lateral_offset * normal_x
    y = sample.y + lateral_offset * normal_y
    return VehicleState(
        x=x,
        y=y,
        yaw=yaw,
        speed=speed,
        vx=speed * math.cos(yaw),
        vy=speed * math.sin(yaw),
        gear=1,
    )
