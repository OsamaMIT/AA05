from __future__ import annotations

import numpy as np

from chrono_a2rl.track.track_loader import create_synthetic_track


def test_projection_on_centerline_has_small_lateral_error() -> None:
    track = create_synthetic_track({"num_points": 240, "radius_x": 60.0, "radius_y": 60.0})
    sample = track.interpolate(25.0)
    projection = track.project_xy(sample.x, sample.y)
    assert np.isclose(projection.n, 0.0, atol=1.0e-6)
    assert projection.on_track


def test_boundary_distances_follow_signed_lateral_error() -> None:
    track = create_synthetic_track(
        {"num_points": 240, "radius_x": 60.0, "radius_y": 60.0, "width_left": 5.0, "width_right": 7.0}
    )
    sample = track.interpolate(0.0)
    left_point = (sample.x - 2.0, sample.y)
    projection = track.project_xy(*left_point)
    assert projection.n > 0.0
    assert np.isclose(projection.distance_left_boundary, 3.0, atol=0.2)
    assert np.isclose(projection.distance_right_boundary, 9.0, atol=0.2)
