from __future__ import annotations

import math

from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig
from chrono_a2rl.vehicle.aero_models import QuadraticAeroModel


def test_aero_drag_and_downforce_increase_with_speed_squared() -> None:
    model = QuadraticAeroModel(A2RLVehicleConfig.load())

    low = model.forces(20.0)
    high = model.forces(40.0)

    assert math.isclose(high.drag_force_n, 4.0 * low.drag_force_n, rel_tol=1e-9)
    assert math.isclose(
        high.downforce_total_n, 4.0 * low.downforce_total_n, rel_tol=1e-9
    )
    assert math.isclose(
        high.front_downforce_n + high.rear_downforce_n,
        high.downforce_total_n,
    )


def test_aero_downforce_safety_clamp_is_enforced() -> None:
    model = QuadraticAeroModel(A2RLVehicleConfig.load())
    assert model.forces(200.0).downforce_total_n == model.max_downforce

