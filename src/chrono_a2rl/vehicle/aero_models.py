"""Quadratic aerodynamic force models."""

from __future__ import annotations

from dataclasses import dataclass

from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig


@dataclass(frozen=True, slots=True)
class AeroForces:
    drag_force_n: float
    downforce_total_n: float
    front_downforce_n: float
    rear_downforce_n: float


class QuadraticAeroModel:
    def __init__(self, config: A2RLVehicleConfig) -> None:
        self.enabled = bool(config.value("aero.enabled", True))
        self.rho = float(config.value("aero.air_density_kg_m3"))
        self.cda = float(config.value("aero.cda_total"))
        self.cla = float(config.value("aero.cla_total"))
        self.front_balance = float(config.value("aero.aero_balance_front"))
        self.rear_balance = float(config.value("aero.aero_balance_rear"))
        self.max_downforce = float(config.value("aero.max_downforce_n"))

    def forces(self, speed_mps: float) -> AeroForces:
        speed_sq = max(float(speed_mps), 0.0) ** 2
        if not self.enabled:
            return AeroForces(0.0, 0.0, 0.0, 0.0)
        drag = 0.5 * self.rho * self.cda * speed_sq
        downforce = min(0.5 * self.rho * self.cla * speed_sq, self.max_downforce)
        return AeroForces(
            drag_force_n=drag,
            downforce_total_n=downforce,
            front_downforce_n=self.front_balance * downforce,
            rear_downforce_n=self.rear_balance * downforce,
        )

