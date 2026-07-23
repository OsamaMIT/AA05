"""Axle tire models with normal-load limits and combined slip."""

from __future__ import annotations

from dataclasses import dataclass
import math

from chrono_a2rl.common.math_utils import clamp
from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig


@dataclass(frozen=True, slots=True)
class TireForceResult:
    longitudinal_force_n: float
    lateral_force_n: float
    longitudinal_limit_n: float
    lateral_limit_n: float
    usage_ratio: float
    slip_angle_rad: float
    effective_mu: float


class AxleTireModel:
    """Linear-slip axle force followed by friction-circle/ellipse saturation."""

    def __init__(
        self,
        *,
        cornering_stiffness_n_rad: float,
        mu_peak: float,
        nominal_load_n: float,
        load_sensitivity: float,
        safety_factor: float,
        mode: str = "friction_ellipse_with_linear_slip",
    ) -> None:
        if mode not in {"friction_circle", "friction_ellipse_with_linear_slip"}:
            raise ValueError(f"Unsupported tire model: {mode}")
        self.cornering_stiffness = float(cornering_stiffness_n_rad)
        self.mu_peak = float(mu_peak)
        self.nominal_load = max(float(nominal_load_n), 1.0)
        self.load_sensitivity = max(float(load_sensitivity), 0.0)
        self.safety_factor = clamp(float(safety_factor), 0.1, 1.0)
        self.mode = mode

    def forces(
        self,
        *,
        normal_load_n: float,
        slip_angle_rad: float,
        longitudinal_request_n: float,
    ) -> TireForceResult:
        fz = max(float(normal_load_n), 1.0)
        load_ratio = fz / self.nominal_load
        mu = self.mu_peak * max(
            0.65,
            1.0 - self.load_sensitivity * max(load_ratio - 1.0, 0.0),
        )
        force_limit = max(mu * fz, 1.0)
        fx = float(longitudinal_request_n)
        fy = self.cornering_stiffness * float(slip_angle_rad)
        if self.mode == "friction_circle":
            norm = math.hypot(fx, fy)
            allowed = self.safety_factor * force_limit
            if norm > allowed:
                scale = allowed / norm
                fx *= scale
                fy *= scale
        else:
            usage = math.hypot(fx / force_limit, fy / force_limit)
            if usage > self.safety_factor:
                scale = self.safety_factor / usage
                fx *= scale
                fy *= scale
        usage = math.hypot(fx / force_limit, fy / force_limit)
        return TireForceResult(
            longitudinal_force_n=fx,
            lateral_force_n=fy,
            longitudinal_limit_n=force_limit,
            lateral_limit_n=force_limit,
            usage_ratio=usage,
            slip_angle_rad=float(slip_angle_rad),
            effective_mu=mu,
        )


def make_axle_tire_models(
    config: A2RLVehicleConfig,
) -> tuple[AxleTireModel, AxleTireModel]:
    mass = float(config.value("vehicle.mass.total_kg"))
    front_fraction = float(config.value("vehicle.mass.front_static_weight_fraction"))
    g = 9.81
    mode = str(config.value("tires.model_default"))
    sensitivity = float(config.value("tires.load_sensitivity.coefficient"))
    safety = float(config.value("tires.combined_slip.safety_factor"))
    front = AxleTireModel(
        cornering_stiffness_n_rad=float(
            config.value("tires.stiffness.cornering_stiffness_front_n_per_rad")
        ),
        mu_peak=float(config.value("tires.friction.mu_peak_front")),
        nominal_load_n=front_fraction * mass * g,
        load_sensitivity=sensitivity,
        safety_factor=safety,
        mode=mode,
    )
    rear = AxleTireModel(
        cornering_stiffness_n_rad=float(
            config.value("tires.stiffness.cornering_stiffness_rear_n_per_rad")
        ),
        mu_peak=float(config.value("tires.friction.mu_peak_rear")),
        nominal_load_n=(1.0 - front_fraction) * mass * g,
        load_sensitivity=sensitivity,
        safety_factor=safety,
        mode=mode,
    )
    return front, rear

