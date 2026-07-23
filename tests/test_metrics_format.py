from __future__ import annotations

import numpy as np

from chrono_a2rl.common.types import EpisodeMetrics
from chrono_a2rl.evaluation.metrics import compute_metrics, format_lap_time, metrics_to_dict


def test_format_lap_time_f1_style() -> None:
    assert format_lap_time(0.0) == "0:00.000"
    assert format_lap_time(83.4564) == "1:23.456"
    assert format_lap_time(427.03999999990157) == "7:07.040"


def test_metrics_accept_f1_formatted_sim_time_logs() -> None:
    rows = [
        {
            "sim_time": "0:12.340",
            "sim_time_seconds": 12.34,
            "speed_kmh": 100.0,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "on_track": True,
        },
        {
            "sim_time": "1:23.456",
            "sim_time_seconds": 83.456,
            "speed_kmh": 100.0,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "on_track": True,
        },
    ]

    metrics = compute_metrics(rows, "max_steps")

    assert np.isclose(metrics.lap_time, 71.116)
    assert metrics.lap_time_formatted == "1:11.116"


def test_episode_time_includes_interval_before_first_logged_step() -> None:
    rows = [
        {
            "sim_time_seconds": 0.02,
            "episode_time_seconds": 0.02,
            "speed_kmh": 100.0,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "on_track": True,
        },
        {
            "sim_time_seconds": 83.46,
            "episode_time_seconds": 83.46,
            "speed_kmh": 100.0,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "on_track": True,
        },
    ]

    metrics = compute_metrics(rows, "lap_completed")

    assert np.isclose(metrics.lap_time, 83.46)
    assert metrics.lap_time_formatted == "1:23.460"


def test_metrics_dict_replaces_lap_time_with_formatted_value() -> None:
    metrics = EpisodeMetrics(
        lap_completed=True,
        lap_time=83.4564,
        lap_time_formatted="1:23.456",
        mean_speed=10.0,
        max_speed=20.0,
    )
    data = metrics_to_dict(metrics)
    assert data["lap_time"] == "1:23.456"
    assert data["lap_time_seconds"] == 83.4564
    assert data["mean_speed_kmh"] == 36.0
    assert data["max_speed_kmh"] == 72.0
    assert "mean_speed" not in data
    assert "max_speed" not in data
    assert "lap_time_formatted" not in data


def test_compute_metrics_reports_speed_policy_collapse_diagnostics() -> None:
    rows = [
        {
            "sim_time": 0.0,
            "speed_kmh": 100.0,
            "target_speed_kmh": 200.0,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "on_track": True,
            "control_saturated": False,
            "action_0": 1.0,
        },
        {
            "sim_time": 0.02,
            "speed_kmh": 120.0,
            "target_speed_kmh": 295.0,
            "lateral_error": 0.1,
            "heading_error": 0.01,
            "on_track": True,
            "control_saturated": False,
            "action_0": 1.35,
        },
    ]

    metrics = compute_metrics(rows, "max_steps")
    data = metrics_to_dict(metrics)

    assert data["mean_speed_scale"] == 1.175
    assert data["min_speed_scale"] == 1.0
    assert data["max_speed_scale"] == 1.35
    assert data["mean_target_speed_kmh"] == 247.5
    assert data["max_target_speed_kmh"] == 295.0
    assert np.isclose(
        data["profile_speed_error_rmse_kmh"],
        np.sqrt((100.0**2 + 175.0**2) / 2.0),
    )
    assert data["profile_speed_error_mae_kmh"] == 137.5


def test_compute_metrics_reports_frontier_corner_and_crash_diagnostics() -> None:
    rows = [
        {
            "sim_time": 0.0,
            "speed_kmh": 120.0,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "on_track": True,
            "validated_progress_m": 300.0,
            "frontier_progress_m": 350.0,
            "frontier_advancement_m": 0.0,
            "frontier_cleared": False,
            "training_role": "start_line",
            "corner_completed": False,
            "corner_score": 0.0,
            "kinetic_crash_penalty": 0.0,
        },
        {
            "sim_time": 0.02,
            "speed_kmh": 140.0,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "on_track": True,
            "validated_progress_m": 400.0,
            "frontier_progress_m": 400.0,
            "frontier_advancement_m": 50.0,
            "frontier_cleared": True,
            "training_role": "start_line",
            "corner_completed": True,
            "corner_score": 550.0,
            "apex_speed_kmh": 130.0,
            "exit_speed_kmh": 145.0,
            "kinetic_crash_penalty": 700.0,
        },
    ]

    data = metrics_to_dict(compute_metrics(rows, "off_track"))

    assert data["max_validated_progress_m"] == 400.0
    assert data["frontier_progress_m"] == 400.0
    assert data["frontier_advancement_m"] == 50.0
    assert data["frontier_cleared"] is True
    assert data["training_role"] == "start_line"
    assert data["corner_completion_count"] == 1
    assert data["mean_corner_score"] == 550.0
    assert data["mean_apex_speed_kmh"] == 130.0
    assert data["mean_exit_speed_kmh"] == 145.0
    assert data["kinetic_crash_penalty"] == 700.0


def test_compute_metrics_reports_longitudinal_policy_diagnostics() -> None:
    rows = [
        {
            "sim_time": 0.0,
            "speed_kmh": 90.0,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "on_track": True,
            "longitudinal_action": 0.8,
            "applied_throttle": 0.6,
            "applied_brake": 0.0,
        },
        {
            "sim_time": 0.02,
            "speed_kmh": 88.0,
            "lateral_error": 0.0,
            "heading_error": 0.0,
            "on_track": True,
            "longitudinal_action": -0.5,
            "applied_throttle": 0.0,
            "applied_brake": 0.4,
        },
    ]

    data = metrics_to_dict(compute_metrics(rows, "max_steps"))

    assert np.isclose(data["mean_longitudinal_action"], 0.15)
    assert data["min_longitudinal_action"] == -0.5
    assert data["max_longitudinal_action"] == 0.8
    assert data["mean_throttle"] == 0.3
    assert data["mean_brake"] == 0.2
    assert data["braking_fraction"] == 0.5
