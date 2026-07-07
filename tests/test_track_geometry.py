from __future__ import annotations

import numpy as np

from chrono_a2rl.track.track_loader import create_synthetic_track, load_tumftm_csv


def test_closed_loop_interpolation_wraps() -> None:
    track = create_synthetic_track(
        {"num_points": 200, "radius_x": 50.0, "radius_y": 50.0, "width_left": 6.0, "width_right": 6.0}
    )
    start = track.interpolate(0.0)
    wrapped = track.interpolate(track.length)
    assert np.isclose(start.x, wrapped.x, atol=1.0e-6)
    assert np.isclose(start.y, wrapped.y, atol=1.0e-6)


def test_curvature_circle_is_positive() -> None:
    radius = 50.0
    track = create_synthetic_track(
        {"num_points": 300, "radius_x": radius, "radius_y": radius, "width_left": 6.0, "width_right": 6.0}
    )
    assert np.all(track.curvature > 0.0)
    assert np.isclose(np.mean(track.curvature), 1.0 / radius, rtol=0.08)


def test_tumftm_comment_header_csv_loads(tmp_path) -> None:
    csv_path = tmp_path / "track.csv"
    csv_path.write_text(
        "# x_m,y_m,w_tr_left_m,w_tr_right_m\n"
        "0.0,0.0,5.0,5.0\n"
        "10.0,0.0,5.0,5.0\n"
        "10.0,10.0,5.0,5.0\n"
        "0.0,10.0,5.0,5.0\n",
        encoding="utf-8",
    )
    track = load_tumftm_csv(csv_path, name="unit_square")
    assert track.centerline.shape == (4, 2)
    assert np.allclose(track.width_left, 5.0)
