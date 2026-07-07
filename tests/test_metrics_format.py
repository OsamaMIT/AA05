from __future__ import annotations

from chrono_a2rl.evaluation.metrics import format_lap_time


def test_format_lap_time_f1_style() -> None:
    assert format_lap_time(0.0) == "0:00.000"
    assert format_lap_time(83.4564) == "1:23.456"
    assert format_lap_time(427.03999999990157) == "7:07.040"
