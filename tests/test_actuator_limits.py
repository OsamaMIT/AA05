from __future__ import annotations

from chrono_a2rl.common.types import VehicleCommand
from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig
from chrono_a2rl.vehicle.actuator_model import A2RLActuatorModel


def test_steering_has_delay_and_rate_limit() -> None:
    model = A2RLActuatorModel(A2RLVehicleConfig.load())
    command = VehicleCommand(steering_target=0.35)

    first = model.update(command, 0.005)
    assert first.steering == 0.0
    for _ in range(5):
        current = model.update(command, 0.005)
    assert 0.0 < current.steering <= 4.0 * 0.030
    assert current.steering < command.steering_target


def test_throttle_brake_conflict_is_removed() -> None:
    model = A2RLActuatorModel(A2RLVehicleConfig.load())
    command = VehicleCommand(throttle_target=0.8, brake_target=1.0)
    result = None
    for _ in range(30):
        result = model.update(command, 0.01)

    assert result is not None
    assert result.throttle == 0.0
    assert result.brake > 0.0
    assert result.saturated

