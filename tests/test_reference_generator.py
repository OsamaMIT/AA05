from __future__ import annotations

import numpy as np

from chrono_a2rl.common.types import TrackState
from chrono_a2rl.control.reference_generator import make_reference, offset_from_fraction
from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_loader import create_synthetic_track


def test_offset_from_fraction_respects_track_margin() -> None:
    track = create_synthetic_track({"width_left": 6.0, "width_right": 8.0})
    assert np.isclose(offset_from_fraction(track, 0.0, 1.0, margin=1.0), 5.0)
    assert np.isclose(offset_from_fraction(track, 0.0, -1.0, margin=1.0), -7.0)


def test_make_reference_carries_lateral_offset() -> None:
    track = create_synthetic_track({"width_left": 6.0, "width_right": 8.0})
    profile = generate_speed_profile(track, {"min_speed": 5.0, "max_speed": 10.0})
    reference = make_reference(
        track,
        profile,
        TrackState(s=0.0),
        lateral_offset=2.5,
    )
    assert reference.target_lateral_offset == 2.5


def test_make_reference_separates_current_control_offset_from_preview_offset() -> None:
    track = create_synthetic_track({"width_left": 6.0, "width_right": 8.0})
    profile = generate_speed_profile(track, {"min_speed": 5.0, "max_speed": 10.0})
    base = make_reference(
        track,
        profile,
        TrackState(s=0.0),
        lateral_offset=1.0,
        lookahead_lateral_offset=1.0,
    )
    preview = make_reference(
        track,
        profile,
        TrackState(s=0.0),
        lateral_offset=1.0,
        lookahead_lateral_offset=3.0,
    )

    assert preview.target_lateral_offset == 1.0
    assert not np.allclose(
        [preview.target_x, preview.target_y],
        [base.target_x, base.target_y],
    )


def test_make_reference_clips_scaled_target_speed_to_profile_cap() -> None:
    track = create_synthetic_track({"width_left": 6.0, "width_right": 8.0})
    profile = generate_speed_profile(
        track,
        {
            "min_speed": 20.0,
            "max_speed": 83.3333333333,
            "max_lateral_accel": 1000.0,
        },
    )

    reference = make_reference(
        track,
        profile,
        TrackState(s=0.0),
        speed_scale=1.5,
    )

    assert np.isclose(reference.target_speed, 83.3333333333)


def test_make_reference_samples_curvature_across_mpc_horizon() -> None:
    track = create_synthetic_track({"radius": 50.0})
    profile = generate_speed_profile(
        track,
        {"min_speed": 10.0, "max_speed": 10.0, "max_lateral_accel": 10.0},
    )

    reference = make_reference(
        track,
        profile,
        TrackState(s=0.0),
        horizon_steps=15,
        control_dt=0.02,
    )

    assert len(reference.curvature_preview) == 15
    assert np.all(np.isfinite(reference.curvature_preview))
