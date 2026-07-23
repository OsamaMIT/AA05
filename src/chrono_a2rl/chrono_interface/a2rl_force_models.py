"""Translate A2RL vehicle telemetry into Chrono force diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from chrono_a2rl.vehicle.telemetry import VehicleTelemetry


@dataclass(frozen=True, slots=True)
class ChassisForceSnapshot:
    longitudinal_force_n: float
    aerodynamic_drag_n: float
    vertical_downforce_n: float
    yaw_rate_rad_s: float


def force_snapshot(telemetry: VehicleTelemetry) -> ChassisForceSnapshot:
    return ChassisForceSnapshot(
        longitudinal_force_n=telemetry.drive_force_n - telemetry.brake_force_n,
        aerodynamic_drag_n=telemetry.drag_force_n,
        vertical_downforce_n=telemetry.downforce_n,
        yaw_rate_rad_s=telemetry.yaw_rate_rad_s,
    )

