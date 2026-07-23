from __future__ import annotations

from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig
from chrono_a2rl.vehicle.powertrain_model import A2RLPowertrainModel


def test_drive_force_is_power_and_rear_grip_limited() -> None:
    model = A2RLPowertrainModel(A2RLVehicleConfig.load())
    output = model.output(
        speed_mps=60.0,
        throttle=1.0,
        rear_normal_load_n=10000.0,
        rear_mu=1.85,
        gear=4,
    )

    assert output.drive_force_n <= output.power_limited_force_n
    assert output.drive_force_n <= output.tire_limited_force_n
    assert output.drive_force_n > 0.0


def test_top_speed_limiter_removes_drive_force() -> None:
    model = A2RLPowertrainModel(A2RLVehicleConfig.load())
    output = model.output(
        speed_mps=model.max_speed,
        throttle=1.0,
        rear_normal_load_n=10000.0,
        rear_mu=1.85,
        gear=6,
    )

    assert output.drive_force_n == 0.0

