from __future__ import annotations

import math

from chrono_a2rl.vehicle.tire_models import AxleTireModel


def _tire() -> AxleTireModel:
    return AxleTireModel(
        cornering_stiffness_n_rad=100000.0,
        mu_peak=1.8,
        nominal_load_n=4000.0,
        load_sensitivity=0.0,
        safety_factor=0.92,
    )


def test_tire_force_is_clamped_by_mu_and_normal_load() -> None:
    result = _tire().forces(
        normal_load_n=4000.0,
        slip_angle_rad=0.3,
        longitudinal_request_n=0.0,
    )

    assert abs(result.lateral_force_n) <= 1.8 * 4000.0 * 0.92
    assert result.usage_ratio <= 0.92 + 1.0e-9


def test_combined_braking_reduces_available_lateral_force() -> None:
    pure_corner = _tire().forces(
        normal_load_n=4000.0,
        slip_angle_rad=0.1,
        longitudinal_request_n=0.0,
    )
    combined = _tire().forces(
        normal_load_n=4000.0,
        slip_angle_rad=0.1,
        longitudinal_request_n=-6500.0,
    )

    assert abs(combined.lateral_force_n) < abs(pure_corner.lateral_force_n)
    assert math.hypot(
        combined.longitudinal_force_n / combined.longitudinal_limit_n,
        combined.lateral_force_n / combined.lateral_limit_n,
    ) <= 0.92 + 1.0e-9

