from __future__ import annotations

import math

from chrono_a2rl.common.types import VehicleCommand, VehicleState
from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig
from chrono_a2rl.vehicle.dynamic_bicycle import DynamicBicycleModel


def _model() -> DynamicBicycleModel:
    return DynamicBicycleModel(A2RLVehicleConfig.load(), physics_dt=0.002)


def test_dynamic_bicycle_state_changes_smoothly_and_steering_is_not_instant() -> None:
    model = _model()
    model.reset(VehicleState(speed=30.0))
    command = VehicleCommand(steering_target=0.20, throttle_target=0.2)

    first = model.step(command, 0.01)
    second = model.step(command, 0.01)
    third = model.step(command, 0.01)

    assert first.steering_angle == 0.0
    assert 0.0 <= second.steering_angle < command.steering_target
    assert third.yaw_rate > second.yaw_rate
    assert math.isfinite(third.x)
    assert abs(third.yaw - second.yaw) < 0.1


def test_full_braking_distance_is_finite_and_plausible() -> None:
    model = _model()
    model.reset(VehicleState(speed=83.333))
    command = VehicleCommand(brake_target=1.0)
    start_x = model.get_state().x

    for _ in range(1000):
        state = model.step(command, 0.01)
        if state.speed <= 80.0 / 3.6:
            break

    distance = state.x - start_x
    assert state.speed <= 80.0 / 3.6
    assert 80.0 < distance < 300.0
    assert model.get_telemetry().combined_slip_usage <= 0.92 + 1.0e-6


def test_vehicle_stays_near_configured_top_speed_under_steady_throttle() -> None:
    model = _model()
    model.reset(VehicleState(speed=20.0))
    command = VehicleCommand(throttle_target=1.0)

    for _ in range(3500):
        state = model.step(command, 0.01)

    assert 285.0 <= state.speed * 3.6 <= 303.0
