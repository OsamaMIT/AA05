from __future__ import annotations

import math

from chrono_a2rl.common.types import TrackState, VehicleCommand, VehicleState
from chrono_a2rl.control.safety_supervisor import SafetySupervisor


def test_safety_supervisor_nan_command_emergency_brakes() -> None:
    supervisor = SafetySupervisor({"max_brake": 1.0}, {})
    safe = supervisor.supervise(
        VehicleCommand(steering_target=math.nan),
        VehicleState(),
        TrackState(on_track=True),
        0.02,
    )
    assert safe.emergency_brake
    assert safe.throttle_target == 0.0
    assert safe.brake_target == 1.0
    assert supervisor.last_reason == "nan_command"


def test_safety_supervisor_resolves_throttle_brake_conflict() -> None:
    supervisor = SafetySupervisor({"max_throttle": 1.0, "max_brake": 1.0}, {})
    safe = supervisor.supervise(
        VehicleCommand(throttle_target=0.6, brake_target=0.3),
        VehicleState(),
        TrackState(on_track=True),
        0.02,
    )
    assert safe.throttle_target > 0.0
    assert safe.brake_target == 0.0
    assert supervisor.last_saturated
