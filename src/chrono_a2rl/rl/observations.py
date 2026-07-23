"""Observation construction for high-level RL policies."""

from __future__ import annotations

import numpy as np

from chrono_a2rl.common.types import ControllerReference, TrackState, VehicleState
from chrono_a2rl.track.speed_profile import SpeedProfile
from chrono_a2rl.track.track_geometry import TrackGeometry


LONGITUDINAL_OBSERVATION_SIZE = 33
PLANNER_OBSERVATION_SIZE = LONGITUDINAL_OBSERVATION_SIZE


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


def make_longitudinal_observation(
    state: VehicleState,
    track_state: TrackState,
    reference: ControllerReference,
    track: TrackGeometry,
    speed_profile: SpeedProfile,
    *,
    previous_pedal_action: float,
    progress_s: float,
    frontier_distance: float = 0.0,
    corner_phase: float = 0.0,
    distance_to_apex: float = 0.0,
    corner_heading_completion: float = 0.0,
    trail_brake_target: float = 0.0,
    curvature_source: str = "raceline",
    lookahead_distances: tuple[float, ...] = (20.0, 40.0, 80.0, 120.0, 180.0),
) -> np.ndarray:
    """Observation for a policy that controls only throttle and braking.

    The path geometry and nominal speed profile are previews, not actuator
    commands. Steering remains entirely under the lateral MPC.
    """

    lookahead_curvatures = [
        track.curvature_at(track_state.s + distance, source=curvature_source)
        for distance in lookahead_distances
    ]
    lookahead_profile_speeds = [
        speed_profile.speed_at(track_state.s + distance)
        for distance in lookahead_distances
    ]
    values = [
        state.speed,
        reference.target_speed,
        track_state.n,
        track_state.heading_error,
        state.yaw_rate,
        state.steering_angle,
        track.curvature_at(track_state.s, source=curvature_source),
        track_state.distance_left_boundary,
        track_state.distance_right_boundary,
        float(track_state.on_curb),
        track_state.curb_penalty_weight,
        state.throttle,
        state.brake,
        previous_pedal_action,
        progress_s / max(track.length, 1.0),
        reference.target_lateral_offset,
        speed_profile.speed_at(track_state.s),
        *lookahead_curvatures,
        *lookahead_profile_speeds,
        track_state.s / max(track.length, 1.0),
        float(np.clip(frontier_distance / max(track.length, 1.0), -1.0, 1.0)),
        corner_phase,
        float(np.clip(distance_to_apex / 300.0, -1.0, 1.0)),
        corner_heading_completion,
        float(np.clip(trail_brake_target, 0.0, 1.0)),
    ]
    return np.asarray(values, dtype=np.float32)
