from __future__ import annotations

import numpy as np
import pandas as pd

from chrono_a2rl.evaluation.profile_optimizer import (
    ProfileRollout,
    _accept_candidate,
    _healthy_segments,
    _profile_from_segment_scales,
    _segment_statistics,
)
from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_loader import create_synthetic_track


def _rollout(**overrides) -> ProfileRollout:
    values = {
        "completed": True,
        "lap_time": 100.0,
        "termination_reason": "lap_completed",
        "progress_m": 1000.0,
        "final_s": 0.0,
        "min_boundary_margin": 0.8,
        "raceline_error_rms": 0.2,
        "max_raceline_error": 1.0,
        "max_heading_error": 0.4,
        "steering_saturation_fraction": 0.01,
        "max_speed_kmh": 250.0,
        "segment_stats": pd.DataFrame(),
    }
    values.update(overrides)
    return ProfileRollout(**values)


def test_segment_statistics_identifies_healthy_repeatable_sections() -> None:
    rows = pd.DataFrame(
        {
            "s": [10.0, 20.0, 110.0, 120.0],
            "speed_kmh": [100.0, 105.0, 80.0, 82.0],
            "boundary_margin": [1.2, 1.1, 0.3, 0.4],
            "raceline_error": [0.1, -0.1, 1.2, 1.0],
            "heading_error": [0.02, 0.03, 0.4, 0.5],
            "pid_mode": ["coast", "throttle", "brake", "brake"],
            "throttle_command": [0.0, 0.3, 0.0, 0.0],
            "brake_command": [0.0, 0.0, 0.4, 0.5],
            "steering_saturated": [False, False, True, True],
        }
    )

    stats = _segment_statistics(rows, 200.0, 100.0, 0.02)
    healthy = _healthy_segments(stats, {}, 2)

    assert healthy.tolist() == [True, False]
    assert np.isclose(stats.loc[0, "elapsed_seconds"], 0.04)
    assert stats.loc[0, "coast_fraction"] == 0.5
    assert stats.loc[1, "braking_fraction"] == 1.0


def test_candidate_requires_faster_completed_lap_with_track_margin() -> None:
    baseline = _rollout()
    faster = _rollout(lap_time=99.5)
    crash = _rollout(
        completed=False,
        lap_time=20.0,
        termination_reason="off_track",
    )
    unsafe = _rollout(lap_time=99.0, min_boundary_margin=0.1)

    assert _accept_candidate(baseline, faster, {}) == (
        True,
        "faster_completed_lap",
    )
    assert _accept_candidate(baseline, crash, {}) == (False, "off_track")
    assert _accept_candidate(baseline, unsafe, {}) == (False, "boundary_margin")


def test_segment_scaling_respects_vehicle_cap_and_longitudinal_limits() -> None:
    track = create_synthetic_track(
        {"num_points": 120, "radius_x": 80.0, "radius_y": 50.0}
    )
    speed_config = {
        "min_speed": 5.0,
        "max_speed": 30.0,
        "max_lateral_accel": 8.0,
        "max_accel": 4.0,
        "max_decel": 8.0,
        "smoothing_window": 1,
    }
    baseline = generate_speed_profile(track, speed_config)
    segment_length = track.length / 4.0

    optimized = _profile_from_segment_scales(
        track,
        baseline,
        np.array([1.05, 1.0, 1.0, 1.0]),
        segment_length,
        speed_config,
        {"scale_smoothing_m": 0.0},
    )

    assert np.any(optimized.speed > baseline.speed)
    assert np.max(optimized.speed) <= 30.0
