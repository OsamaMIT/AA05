"""Placeholders for future disturbance robustness tests."""

from __future__ import annotations


def planned_disturbance_suite() -> list[str]:
    """Return planned disturbance categories."""

    return ["crosswind", "friction_drop", "actuator_delay", "sensor_noise"]
