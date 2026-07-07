from __future__ import annotations

from chrono_a2rl.chrono_interface.direct_backend import MockChronoBackend
from chrono_a2rl.common.types import VehicleCommand, VehicleState


def test_mock_backend_steps_forward() -> None:
    backend = MockChronoBackend(
        {
            "wheelbase": 3.0,
            "max_accel": 5.0,
            "max_decel": 8.0,
            "drag_coefficient": 0.0,
            "rolling_resistance": 0.0,
            "throttle_time_constant": 0.0,
            "brake_time_constant": 0.0,
            "steering_time_constant": 0.0,
        }
    )
    backend.reset(VehicleState(speed=1.0))
    state = backend.step(VehicleCommand(throttle_target=1.0), 0.1)
    assert state.sim_time == 0.1
    assert state.speed > 1.0
    assert state.x > 0.0
