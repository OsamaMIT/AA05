from __future__ import annotations

import numpy as np

from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_geometry import TrackGeometry
from chrono_a2rl.track.track_loader import create_synthetic_track


def test_speed_profile_is_clamped() -> None:
    track = create_synthetic_track({"num_points": 200, "radius_x": 30.0, "radius_y": 30.0})
    profile = generate_speed_profile(
        track,
        {
            "min_speed": 4.0,
            "max_speed": 12.0,
            "max_lateral_accel": 3.0,
            "smoothing_window": 5,
        },
    )
    assert np.min(profile.speed) >= 4.0
    assert np.max(profile.speed) <= 12.0
    assert profile.speed_at(track.length + 1.0) == profile.speed_at(1.0)


def test_speed_profile_can_follow_raceline_curvature() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    centerline = np.column_stack([50.0 * np.cos(angles), 50.0 * np.sin(angles)])
    raceline = np.column_stack([30.0 * np.cos(angles), 30.0 * np.sin(angles)])
    track = TrackGeometry(
        centerline,
        width_left=25.0,
        width_right=25.0,
        raceline=raceline,
    )
    common = {
        "min_speed": 1.0,
        "max_speed": 100.0,
        "max_lateral_accel": 10.0,
        "smoothing_window": 1,
    }

    centerline_profile = generate_speed_profile(
        track,
        {**common, "curvature_source": "centerline"},
    )
    raceline_profile = generate_speed_profile(
        track,
        {**common, "curvature_source": "raceline"},
    )

    assert np.mean(raceline_profile.speed) < np.mean(centerline_profile.speed)
    assert np.isclose(np.mean(raceline_profile.speed), np.sqrt(300.0), rtol=0.08)


def test_speed_profile_loads_optimized_kmh_artifact(tmp_path) -> None:
    track = create_synthetic_track(
        {"num_points": 100, "radius_x": 40.0, "radius_y": 40.0}
    )
    profile_path = tmp_path / "optimized.csv"
    profile_path.write_text(
        "s_m,speed_kmh\n"
        f"0.0,36.0\n"
        f"{0.5 * track.length},72.0\n",
        encoding="utf-8",
    )

    profile = generate_speed_profile(
        track,
        {
            "profile_path": str(profile_path),
            "min_speed": 0.0,
            "max_speed": 15.0,
        },
    )

    assert np.isclose(profile.speed_at(0.0), 10.0)
    assert np.isclose(profile.speed_at(0.5 * track.length), 15.0)
    assert np.max(profile.speed) <= 15.0


def test_shallow_curve_envelope_raises_only_shallow_bend_speed() -> None:
    common = {
        "min_speed": 1.0,
        "max_speed": 100.0,
        "max_lateral_accel": 19.0,
        "smoothing_window": 1,
    }
    envelope = {
        "shallow_curve_max_lateral_accel": 27.0,
        "shallow_curve_curvature_full": 0.006,
        "shallow_curve_curvature_end": 0.012,
    }
    shallow_track = create_synthetic_track(
        {"num_points": 400, "radius_x": 120.0, "radius_y": 120.0}
    )
    tight_track = create_synthetic_track(
        {"num_points": 400, "radius_x": 30.0, "radius_y": 30.0}
    )

    shallow_base = generate_speed_profile(shallow_track, common)
    shallow_raised = generate_speed_profile(shallow_track, {**common, **envelope})
    tight_base = generate_speed_profile(tight_track, common)
    tight_raised = generate_speed_profile(tight_track, {**common, **envelope})

    assert np.mean(shallow_raised.speed) > 1.10 * np.mean(shallow_base.speed)
    assert np.allclose(tight_raised.speed, tight_base.speed, rtol=0.01)
