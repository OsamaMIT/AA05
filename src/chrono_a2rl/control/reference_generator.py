"""Controller reference generation."""

from __future__ import annotations

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
) -> ControllerReference:
    """Build a local controller reference from track state and speed profile."""

    target_speed = speed_profile.speed_at(track_state.s) * speed_scale
    lookahead_s = track_state.s + max(target_speed, 0.0) * lookahead_time
    sample = track.interpolate(lookahead_s)
    return ControllerReference(
        target_speed=target_speed,
        target_s=sample.s,
        target_x=sample.x,
        target_y=sample.y,
        target_yaw=sample.heading,
        target_curvature=sample.curvature,
    )
