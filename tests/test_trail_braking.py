from __future__ import annotations

import numpy as np

from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.common.types import VehicleState
from chrono_a2rl.rl.corner_progress import CornerProgressTracker
from chrono_a2rl.rl.trail_braking import compute_trail_braking_reference
from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_loader import load_track_from_config


def _components():
    config = load_experiment_config("configs/experiments/rl_planner_yas_marina.yaml")
    track = load_track_from_config(config["track"])
    profile = generate_speed_profile(track, config["speed_profile"])
    tracker = CornerProgressTracker(track, profile, config["reward"])
    return config, track, profile, tracker


def _reference_at(s: float, speed: float):
    config, track, profile, tracker = _components()
    point = track.interpolate(s)
    state = VehicleState(
        x=point.x,
        y=point.y,
        yaw=point.heading,
        speed=speed,
    )
    track_state = track.track_state_at_pose(point.x, point.y, point.heading)
    tracker.reset(state, track_state)
    reference = compute_trail_braking_reference(
        tracker=tracker,
        speed_profile=profile,
        s=s,
        speed=speed,
        vehicle_config=config["vehicle"],
        config=config["reward"],
    )
    return reference


def test_trail_reference_is_inactive_outside_lookahead_and_after_apex() -> None:
    clear_straight = _reference_at(1100.0, 81.0)
    after_apex = _reference_at(1530.0, 81.0)

    assert not clear_straight.active
    assert clear_straight.target_brake == 0.0
    assert not after_apex.active
    assert after_apex.target_brake == 0.0
    assert after_apex.phase == 1.0


def test_yas_marina_low_speed_complex_requests_brake_before_entry() -> None:
    reference = _reference_at(1230.0, 81.0)

    assert reference.corner_id == 2
    assert reference.active
    assert reference.distance_to_entry > 0.0
    assert 0.0 < reference.target_brake <= 0.70


def test_trail_reference_tapers_monotonically_from_entry_to_apex() -> None:
    targets = [
        _reference_at(s, 81.0).target_brake
        for s in (1280.0, 1320.0, 1400.0, 1480.0, 1520.0)
    ]

    assert targets[0] <= 0.70
    assert all(left >= right for left, right in zip(targets, targets[1:]))
    assert targets[-1] < 0.02
    assert np.all(np.asarray(targets) >= 0.0)
