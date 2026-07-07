"""Frenet coordinate helpers."""

from __future__ import annotations

from chrono_a2rl.common.types import TrackState
from chrono_a2rl.track.track_geometry import FrenetProjection, TrackGeometry


def project_xy(track: TrackGeometry, x: float, y: float) -> FrenetProjection:
    """Project a point onto a track."""

    return track.project_xy(x, y)


def track_state_from_pose(track: TrackGeometry, x: float, y: float, yaw: float) -> TrackState:
    """Compute TrackState from a world pose."""

    return track.track_state_at_pose(x, y, yaw)
