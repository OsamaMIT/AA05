"""Drive-by-wire command shaping and actuator limits."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from chrono_a2rl.common.math_utils import clamp
from chrono_a2rl.common.types import VehicleCommand, VehicleState


class DBWModel:
    """Convert target commands into safe applied actuator commands."""

    def __init__(self, vehicle_config: dict[str, Any] | None = None) -> None:
        cfg = vehicle_config or {}
        self.max_steer = float(cfg.get("max_steer", 0.45))
        self.max_steer_rate = float(cfg.get("max_steer_rate", 1.2))
        self.max_throttle = float(cfg.get("max_throttle", 1.0))
        self.max_brake = float(cfg.get("max_brake", 1.0))
        self.steering_tau = float(cfg.get("steering_time_constant", 0.06))
        self.throttle_tau = float(cfg.get("throttle_time_constant", 0.10))
        self.brake_tau = float(cfg.get("brake_time_constant", 0.07))
        self._last_command = VehicleCommand()
        self.last_saturated = False

    def reset(self) -> None:
        """Reset actuator memory."""

        self._last_command = VehicleCommand()
        self.last_saturated = False

    def apply(self, command: VehicleCommand, state: VehicleState, dt: float) -> VehicleCommand:
        """Return the actuator-limited command applied to the vehicle."""

        if dt <= 0.0:
            dt = 1.0e-3
        self.last_saturated = False

        if command.emergency_brake:
            applied = replace(
                command,
                steering_target=clamp(command.steering_target, -self.max_steer, self.max_steer),
                throttle_target=0.0,
                brake_target=self.max_brake,
                emergency_brake=True,
            )
            self._last_command = applied
            return applied

        steering_target = clamp(command.steering_target, -self.max_steer, self.max_steer)
        throttle_target = clamp(command.throttle_target, 0.0, self.max_throttle)
        brake_target = clamp(command.brake_target, 0.0, self.max_brake)

        self.last_saturated = (
            not math.isclose(steering_target, command.steering_target)
            or not math.isclose(throttle_target, command.throttle_target)
            or not math.isclose(brake_target, command.brake_target)
        )

        if throttle_target > 0.0 and brake_target > 0.0:
            self.last_saturated = True
            if brake_target >= throttle_target:
                throttle_target = 0.0
            else:
                brake_target = 0.0

        max_delta = self.max_steer_rate * dt
        rate_limited_steer = clamp(
            steering_target,
            state.steering_angle - max_delta,
            state.steering_angle + max_delta,
        )
        if not math.isclose(rate_limited_steer, steering_target):
            self.last_saturated = True

        steering = self._first_order_lag(
            previous=state.steering_angle,
            target=rate_limited_steer,
            dt=dt,
            tau=self.steering_tau,
        )
        throttle = self._first_order_lag(
            previous=state.throttle,
            target=throttle_target,
            dt=dt,
            tau=self.throttle_tau,
        )
        brake = self._first_order_lag(
            previous=state.brake,
            target=brake_target,
            dt=dt,
            tau=self.brake_tau,
        )

        if throttle > 0.01 and brake > 0.01:
            self.last_saturated = True
            if brake >= throttle:
                throttle = 0.0
            else:
                brake = 0.0

        applied = replace(
            command,
            steering_target=clamp(steering, -self.max_steer, self.max_steer),
            throttle_target=clamp(throttle, 0.0, self.max_throttle),
            brake_target=clamp(brake, 0.0, self.max_brake),
            gear_request=max(1, int(command.gear_request)),
        )
        self._last_command = applied
        return applied

    @staticmethod
    def _first_order_lag(previous: float, target: float, dt: float, tau: float) -> float:
        if tau <= 0.0:
            return target
        alpha = clamp(dt / (tau + dt), 0.0, 1.0)
        return float(previous + alpha * (target - previous))
