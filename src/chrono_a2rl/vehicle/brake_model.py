"""Axle-level carbon brake force requests."""

from __future__ import annotations

from dataclasses import dataclass

from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig


@dataclass(frozen=True, slots=True)
class BrakeForces:
    front_request_n: float
    rear_request_n: float
    total_request_n: float


class A2RLBrakeModel:
    def __init__(self, config: A2RLVehicleConfig) -> None:
        self.mass = float(config.value("vehicle.mass.total_kg"))
        self.max_decel = float(config.value("brakes.max_decel_mps2"))
        self.front_bias = float(config.value("brakes.brake_bias_front"))

    def force_request(self, brake: float) -> BrakeForces:
        total = max(0.0, min(float(brake), 1.0)) * self.mass * self.max_decel
        return BrakeForces(
            front_request_n=self.front_bias * total,
            rear_request_n=(1.0 - self.front_bias) * total,
            total_request_n=total,
        )

