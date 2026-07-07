from __future__ import annotations

import math

import numpy as np

from chrono_a2rl.common.config import load_yaml
from chrono_a2rl.evaluation.metrics import compute_metrics
from chrono_a2rl.track.track_loader import load_track_from_config


def test_processed_yas_marina_tumftm_track_loads() -> None:
    config = load_yaml("configs/track/yas_marina.yaml")
    track = load_track_from_config(config)

    assert track.name == "yas_marina"
    assert len(track.centerline) == 1110
    assert track.raceline is not None
    assert len(track.raceline) == 1095
    assert np.isclose(track.length, 5546.57, atol=0.5)
    assert np.isclose(float(np.min(track.width_right)), 4.862, atol=0.001)
    assert np.isclose(float(np.min(track.width_left)), 4.559, atol=0.001)
    assert np.all(track.width_left > 0.0)
    assert np.all(track.width_right > 0.0)


def test_level1_curbs_live_near_track_limits() -> None:
    config = load_yaml("configs/track/yas_marina.yaml")
    track = load_track_from_config(config)
    sample = track.interpolate(120.0)
    normal_left = np.array([-math.sin(sample.heading), math.cos(sample.heading)])
    center = np.array([sample.x, sample.y])

    center_state = track.track_state_at_pose(sample.x, sample.y, sample.heading)
    assert center_state.on_track
    assert not center_state.on_curb

    left_point = center + normal_left * (sample.width_left - 0.5)
    left_state = track.track_state_at_pose(float(left_point[0]), float(left_point[1]), sample.heading)
    assert left_state.on_track
    assert left_state.on_curb
    assert left_state.curb_side == "left"
    assert np.isclose(left_state.curb_penalty_weight, 0.2)

    right_point = center - normal_left * (sample.width_right - 0.5)
    right_state = track.track_state_at_pose(float(right_point[0]), float(right_point[1]), sample.heading)
    assert right_state.on_track
    assert right_state.on_curb
    assert right_state.curb_side == "right"
    assert np.isclose(right_state.curb_penalty_weight, 0.2)

    outside_left = center + normal_left * (sample.width_left + 0.2)
    outside_state = track.track_state_at_pose(
        float(outside_left[0]),
        float(outside_left[1]),
        sample.heading,
    )
    assert not outside_state.on_track


def test_metrics_include_curb_usage() -> None:
    rows = [
        {
            "sim_time": 0.0,
            "speed": 10.0,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "on_track": True,
            "on_curb": False,
            "curb_penalty_weight": 0.0,
            "control_saturated": False,
        },
        {
            "sim_time": 0.02,
            "speed": 11.0,
            "lateral_error": 0.2,
            "heading_error": 0.01,
            "on_track": True,
            "on_curb": True,
            "curb_penalty_weight": 0.2,
            "control_saturated": False,
        },
    ]
    metrics = compute_metrics(rows, "timeout")
    assert metrics.curb_sample_count == 1
    assert np.isclose(metrics.curb_usage_fraction, 0.5)
    assert np.isclose(metrics.curb_penalty_total, 0.2)
    assert metrics.lap_time_formatted == "0:00.020"
