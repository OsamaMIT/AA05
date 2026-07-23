from __future__ import annotations

import numpy as np

from chrono_a2rl.track.track_geometry import TrackGeometry
from chrono_a2rl.track.track_loader import create_synthetic_track, load_tumftm_csv


def test_closed_loop_interpolation_wraps() -> None:
    track = create_synthetic_track(
        {"num_points": 200, "radius_x": 50.0, "radius_y": 50.0, "width_left": 6.0, "width_right": 6.0}
    )
    start = track.interpolate(0.0)
    wrapped = track.interpolate(track.length)
    assert np.isclose(start.x, wrapped.x, atol=1.0e-6)
    assert np.isclose(start.y, wrapped.y, atol=1.0e-6)


def test_start_finish_crossing_requires_forward_wrap() -> None:
    track = create_synthetic_track(
        {"num_points": 200, "radius_x": 50.0, "radius_y": 50.0}
    )

    assert track.crossed_line_forward(track.length - 1.0, 1.0)
    assert not track.crossed_line_forward(1.0, track.length - 1.0)
    assert not track.crossed_line_forward(
        0.97 * track.length,
        0.98 * track.length,
    )


def test_start_finish_crossing_supports_configured_line_position() -> None:
    track = create_synthetic_track(
        {"num_points": 200, "radius_x": 50.0, "radius_y": 50.0}
    )
    line_s = 100.0

    assert track.crossed_line_forward(line_s - 1.0, line_s + 1.0, line_s=line_s)
    assert not track.crossed_line_forward(line_s + 1.0, line_s - 1.0, line_s=line_s)


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


def test_raceline_lateral_offset_interpolates_closed_loop() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    centerline = np.column_stack([50.0 * np.cos(angles), 50.0 * np.sin(angles)])
    raceline = np.column_stack([48.0 * np.cos(angles), 48.0 * np.sin(angles)])
    track = TrackGeometry(
        centerline,
        width_left=6.0,
        width_right=6.0,
        raceline=raceline,
    )

    assert np.isclose(track.raceline_lateral_offset_at(0.0), 2.0, atol=0.1)
    assert np.isclose(
        track.raceline_lateral_offset_at(track.length),
        track.raceline_lateral_offset_at(0.0),
        atol=1.0e-6,
    )


def test_raceline_curvature_uses_green_path_geometry() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    centerline = np.column_stack([50.0 * np.cos(angles), 50.0 * np.sin(angles)])
    raceline = np.column_stack([40.0 * np.cos(angles), 40.0 * np.sin(angles)])
    track = TrackGeometry(
        centerline,
        width_left=15.0,
        width_right=15.0,
        raceline=raceline,
    )

    samples = [track.raceline_curvature_at(s) for s in np.linspace(0.0, track.length, 20)]

    assert np.isclose(np.mean(samples), 1.0 / 40.0, rtol=0.08)
    assert np.isclose(
        track.raceline_curvature_at(track.length),
        track.raceline_curvature_at(0.0),
    )
