from __future__ import annotations

import numpy as np

from chrono_a2rl.chrono_interface.reset_manager import initial_state_from_track
from chrono_a2rl.rl.reset_sampling import sample_initial_state
from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_geometry import TrackGeometry
from chrono_a2rl.track.track_loader import create_synthetic_track


def test_initial_state_from_track_supports_lateral_and_heading_offsets() -> None:
    track = create_synthetic_track({"width_left": 6.0, "width_right": 6.0})
    state = initial_state_from_track(
        track,
        s=10.0,
        speed=5.0,
        lateral_offset=1.2,
        heading_error=0.1,
    )
    track_state = track.track_state_at_pose(state.x, state.y, state.yaw)

    assert np.isclose(track_state.n, 1.2, atol=1.0e-2)
    assert np.isclose(track_state.heading_error, 0.1, atol=1.0e-2)


def test_sample_initial_state_randomizes_episode_start() -> None:
    track = create_synthetic_track({"width_left": 6.0, "width_right": 6.0})
    simulation = {"initial_speed": 3.0}
    rl = {
        "randomize_resets": True,
        "reset_speed_min": 8.0,
        "reset_speed_max": 12.0,
        "reset_lateral_offset_max": 1.0,
        "reset_heading_error_max": 0.05,
    }
    state_a = sample_initial_state(
        track=track,
        simulation_config=simulation,
        rl_config=rl,
        rng=np.random.default_rng(1),
    )
    state_b = sample_initial_state(
        track=track,
        simulation_config=simulation,
        rl_config=rl,
        rng=np.random.default_rng(2),
    )

    assert 8.0 <= state_a.speed <= 12.0
    assert 8.0 <= state_b.speed <= 12.0
    assert not np.isclose(state_a.x, state_b.x)


def test_sample_initial_state_can_reset_around_raceline() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    centerline = np.column_stack([50.0 * np.cos(angles), 50.0 * np.sin(angles)])
    raceline = np.column_stack([48.0 * np.cos(angles), 48.0 * np.sin(angles)])
    track = TrackGeometry(centerline, width_left=6.0, width_right=6.0, raceline=raceline)
    state = sample_initial_state(
        track=track,
        simulation_config={"initial_speed": 5.0},
        rl_config={"randomize_resets": False, "lateral_offset_reference": "raceline"},
        rng=np.random.default_rng(3),
    )
    track_state = track.track_state_at_pose(state.x, state.y, state.yaw)

    assert np.isclose(track_state.n, track.raceline_lateral_offset_at(0.0), atol=0.1)
    expected_yaw = track.raceline_heading_at(0.0)
    assert np.isclose(state.yaw, expected_yaw, atol=1.0e-6)


def test_sample_initial_state_can_use_profile_speed_mode() -> None:
    track = create_synthetic_track({"width_left": 6.0, "width_right": 6.0})
    profile = generate_speed_profile(
        track,
        {
            "min_speed": 20.0,
            "max_speed": 40.0,
            "max_lateral_accel": 1000.0,
        },
    )
    state = sample_initial_state(
        track=track,
        simulation_config={"initial_speed": 5.0},
        rl_config={
            "randomize_resets": True,
            "reset_s_mode": "fixed",
            "reset_s": 0.0,
            "reset_speed_mode": "profile",
            "reset_speed_scale_min": 1.0,
            "reset_speed_scale_max": 1.0,
        },
        rng=np.random.default_rng(4),
        speed_profile=profile,
        vehicle_config={"max_speed": 30.0},
    )

    assert np.isclose(state.speed, 30.0)
