"""Axle normal-load and simplified load-transfer approximation."""

from __future__ import annotations

from dataclasses import dataclass

from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig
from chrono_a2rl.vehicle.aero_models import AeroForces


@dataclass(frozen=True, slots=True)
class AxleLoads:
    front_normal_load_n: float
    rear_normal_load_n: float
    longitudinal_transfer_n: float
    lateral_transfer_front_n: float
    lateral_transfer_rear_n: float


class SuspensionLoadTransferModel:
    def __init__(self, config: A2RLVehicleConfig) -> None:
        self.mass = float(config.value("vehicle.mass.total_kg"))
        self.front_fraction = float(
            config.value("vehicle.mass.front_static_weight_fraction")
        )
        self.wheelbase = float(config.value("geometry.wheelbase_m"))
        self.cg_height = float(config.value("geometry.cg_height_m"))
        self.front_track = float(config.value("geometry.front_track_m"))
        self.rear_track = float(config.value("geometry.rear_track_m"))
        self.roll_front = float(
            config.value("suspension.load_transfer.roll_stiffness_front_fraction")
        )
        self.roll_rear = float(
            config.value("suspension.load_transfer.roll_stiffness_rear_fraction")
        )
        self.g = 9.81

    def loads(
        self,
        *,
        longitudinal_accel_mps2: float,
        lateral_accel_mps2: float,
        aero: AeroForces,
    ) -> AxleLoads:
        static_front = self.front_fraction * self.mass * self.g
        static_rear = (1.0 - self.front_fraction) * self.mass * self.g
        transfer = (
            self.mass
            * float(longitudinal_accel_mps2)
            * self.cg_height
            / max(self.wheelbase, 1.0e-6)
        )
        front = static_front + aero.front_downforce_n - transfer
        rear = static_rear + aero.rear_downforce_n + transfer
        lateral_total = (
            self.mass * abs(float(lateral_accel_mps2)) * self.cg_height
        )
        front_lateral = (
            self.roll_front * lateral_total / max(self.front_track, 1.0e-6)
        )
        rear_lateral = (
            self.roll_rear * lateral_total / max(self.rear_track, 1.0e-6)
        )
        return AxleLoads(
            front_normal_load_n=max(front, 100.0),
            rear_normal_load_n=max(rear, 100.0),
            longitudinal_transfer_n=transfer,
            lateral_transfer_front_n=front_lateral,
            lateral_transfer_rear_n=rear_lateral,
        )
