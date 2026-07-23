"""Power- and rear-tire-limited EAV24-style drivetrain."""

from __future__ import annotations

from dataclasses import dataclass

from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig


@dataclass(frozen=True, slots=True)
class PowertrainOutput:
    drive_force_n: float
    power_limited_force_n: float
    tire_limited_force_n: float
    rolling_resistance_n: float
    gear: int


class A2RLPowertrainModel:
    def __init__(self, config: A2RLVehicleConfig) -> None:
        self.power_w = float(config.value("powertrain.engine.max_power_kw")) * 1000.0
        self.efficiency = float(
            config.value("powertrain.drivetrain.drivetrain_efficiency")
        )
        self.max_speed = float(
            config.value("powertrain.longitudinal_limits.max_speed_mps")
        )
        self.max_low_accel = float(
            config.value("powertrain.longitudinal_limits.max_accel_low_speed_mps2")
        )
        self.mass = float(config.value("vehicle.mass.total_kg"))
        self.rolling_coefficient = float(
            config.value("powertrain.rolling_resistance_coefficient")
        )
        self.gear_ratios = tuple(
            float(value)
            for value in config.value("powertrain.gearbox.gear_ratios")["values"]
        )
        self.auto_shift = bool(config.value("powertrain.gearbox.auto_shift.enabled"))

    def output(
        self,
        *,
        speed_mps: float,
        throttle: float,
        rear_normal_load_n: float,
        rear_mu: float,
        gear: int,
    ) -> PowertrainOutput:
        speed = max(float(speed_mps), 0.0)
        pedal = max(0.0, min(float(throttle), 1.0))
        power_force = self.power_w * self.efficiency / max(speed, 12.0)
        low_speed_force = self.mass * self.max_low_accel
        power_force = min(power_force, low_speed_force)
        tire_force = max(float(rear_mu) * float(rear_normal_load_n), 0.0)
        limiter = max(0.0, min((self.max_speed - speed) / 2.0, 1.0))
        drive = pedal * min(power_force, tire_force) * limiter
        rolling = self.rolling_coefficient * self.mass * 9.81 if speed > 0.1 else 0.0
        selected_gear = self._automatic_gear(speed) if self.auto_shift else max(1, gear)
        return PowertrainOutput(
            drive_force_n=drive,
            power_limited_force_n=power_force,
            tire_limited_force_n=tire_force,
            rolling_resistance_n=rolling,
            gear=selected_gear,
        )

    def _automatic_gear(self, speed: float) -> int:
        fraction = max(0.0, min(speed / max(self.max_speed, 1.0), 0.999))
        return min(6, 1 + int(fraction * 6.0))
