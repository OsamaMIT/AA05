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
