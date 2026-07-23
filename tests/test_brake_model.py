from __future__ import annotations

import math

from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig
from chrono_a2rl.vehicle.brake_model import A2RLBrakeModel


def test_brake_request_uses_bias_and_configured_max_decel() -> None:
    config = A2RLVehicleConfig.load()
    model = A2RLBrakeModel(config)
    result = model.force_request(1.0)

    assert math.isclose(result.total_request_n, 690.0 * 20.0)
    assert math.isclose(result.front_request_n / result.total_request_n, 0.58)
    assert math.isclose(
        result.front_request_n + result.rear_request_n, result.total_request_n
    )

