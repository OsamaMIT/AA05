"""Initial-state helpers for closed-loop simulations."""

from __future__ import annotations

import math

from chrono_a2rl.common.types import VehicleState
from chrono_a2rl.track.track_geometry import TrackGeometry


def initial_state_from_track(
    track: TrackGeometry,
    *,
    s: float = 0.0,
    speed: float = 0.0,
) -> VehicleState:
    """Create an initial VehicleState aligned with the track centerline."""

    sample = track.interpolate(s)
    return VehicleState(
        x=sample.x,
        y=sample.y,
        yaw=sample.heading,
        speed=speed,
        vx=speed * math.cos(sample.heading),
        vy=speed * math.sin(sample.heading),
        gear=1,
    )
