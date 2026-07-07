from __future__ import annotations

import numpy as np

from chrono_a2rl.track.speed_profile import generate_speed_profile
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
