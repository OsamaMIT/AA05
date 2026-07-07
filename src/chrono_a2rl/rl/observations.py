"""Observation construction for high-level RL policies."""

from __future__ import annotations

import numpy as np

from chrono_a2rl.common.types import ControllerReference, TrackState, VehicleState


def make_observation(
    state: VehicleState,
    track_state: TrackState,
    reference: ControllerReference,
    *,
    previous_action: float,
    progress_s: float,
) -> np.ndarray:
    """Create the initial fixed observation vector."""

    return np.array(
        [
            state.speed,
            reference.target_speed,
            track_state.n,
            track_state.heading_error,
            state.yaw_rate,
            state.steering_angle,
            track_state.curvature,
            track_state.distance_left_boundary,
            track_state.distance_right_boundary,
            previous_action,
            progress_s,
        ],
        dtype=np.float32,
    )
