"""Physics telemetry emitted by A2RL-style vehicle models."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class VehicleTelemetry:
    speed_mps: float = 0.0
    speed_kmh: float = 0.0
    longitudinal_accel_mps2: float = 0.0
    lateral_accel_mps2: float = 0.0
    yaw_rate_rad_s: float = 0.0
    slip_angle_front_rad: float = 0.0
    slip_angle_rear_rad: float = 0.0
    tire_usage_front: float = 0.0
    tire_usage_rear: float = 0.0
    combined_slip_usage: float = 0.0
    front_normal_load_n: float = 0.0
    rear_normal_load_n: float = 0.0
    downforce_n: float = 0.0
    drag_force_n: float = 0.0
    drive_force_n: float = 0.0
    brake_force_n: float = 0.0
    steering_target_rad: float = 0.0
    steering_actual_rad: float = 0.0
    throttle_target: float = 0.0
    throttle_actual: float = 0.0
    brake_target: float = 0.0
    brake_actual: float = 0.0
    gear: int = 1
    off_track: bool = False
    on_curb: bool = False
    brake_temp_estimate_c: float = 20.0

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)

