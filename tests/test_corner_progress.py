from __future__ import annotations

from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.rl.corner_progress import build_corner_segments
from chrono_a2rl.track.track_loader import load_track_from_config


def test_yas_marina_corner_segments_include_turn_one() -> None:
    config = load_experiment_config("configs/experiments/rl_planner_yas_marina.yaml")
    track = load_track_from_config(config["track"])

    segments = build_corner_segments(track)
    turn_one = segments[0]

    assert len(segments) >= 10
    assert 350.0 < turn_one.entry_s < 390.0
    assert 400.0 < turn_one.apex_s < 430.0
    assert 450.0 < turn_one.exit_s < 490.0
    assert turn_one.length >= 90.0
    assert turn_one.expected_heading_change > 1.0
