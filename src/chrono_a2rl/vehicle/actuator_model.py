"""Delayed, rate-limited first-order drive-by-wire actuators."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from chrono_a2rl.common.math_utils import clamp
from chrono_a2rl.common.types import VehicleCommand
from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig


class DelayedFirstOrderActuator:
    """Scalar actuator with transport delay, rate limit, and first-order lag."""

    def __init__(
        self,
        *,
        minimum: float,
        maximum: float,
        rate_limit_per_s: float,
        delay_s: float,
        time_constant_s: float,
    ) -> None:
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.rate_limit = max(float(rate_limit_per_s), 0.0)
        self.delay_s = max(float(delay_s), 0.0)
        self.time_constant = max(float(time_constant_s), 0.0)
        self.value = 0.0
        self._clock = 0.0
        self._history: deque[tuple[float, float]] = deque([(0.0, 0.0)])

    def reset(self, value: float = 0.0) -> None:
        self.value = clamp(float(value), self.minimum, self.maximum)
        self._clock = 0.0
        self._history = deque([(0.0, self.value)])

    def update(self, target: float, dt: float) -> float:
        dt_safe = max(float(dt), 1.0e-6)
        self._clock += dt_safe
        clipped_target = clamp(float(target), self.minimum, self.maximum)
        self._history.append((self._clock, clipped_target))
        delayed_time = self._clock - self.delay_s
        delayed_target = self._history[0][1]
        while len(self._history) > 1 and self._history[1][0] <= delayed_time:
            self._history.popleft()
            delayed_target = self._history[0][1]
        alpha = (
            1.0
            if self.time_constant <= 0.0
            else 1.0 - math.exp(-dt_safe / self.time_constant)
        )
        lagged = self.value + alpha * (delayed_target - self.value)
        max_change = self.rate_limit * dt_safe
        if self.rate_limit > 0.0:
            lagged = clamp(lagged, self.value - max_change, self.value + max_change)
        self.value = clamp(lagged, self.minimum, self.maximum)
        return self.value


@dataclass(frozen=True, slots=True)
class AppliedActuators:
    steering: float
    throttle: float
    brake: float
    gear: int
    saturated: bool


class A2RLActuatorModel:
    """Coupled steering, pedal, and gear actuators."""

    def __init__(self, config: A2RLVehicleConfig) -> None:
        self.max_steer = float(config.value("actuators.steering.max_angle_rad"))
        self.steering = DelayedFirstOrderActuator(
            minimum=-self.max_steer,
            maximum=self.max_steer,
            rate_limit_per_s=float(
                config.value("actuators.steering.max_rate_rad_s")
            ),
            delay_s=float(config.value("actuators.steering.delay_s")),
            time_constant_s=float(
                config.value("actuators.steering.first_order_response_time_s")
            ),
        )
        self.throttle = DelayedFirstOrderActuator(
            minimum=float(config.value("actuators.throttle.min_value")),
            maximum=float(config.value("actuators.throttle.max_value")),
            rate_limit_per_s=float(
                config.value("actuators.throttle.rate_limit_per_s")
            ),
            delay_s=float(config.value("actuators.throttle.delay_s")),
            time_constant_s=float(
                config.value("actuators.throttle.first_order_response_time_s")
            ),
        )
        self.brake = DelayedFirstOrderActuator(
            minimum=float(config.value("actuators.brake.min_value")),
            maximum=float(config.value("actuators.brake.max_value")),
            rate_limit_per_s=float(config.value("actuators.brake.rate_limit_per_s")),
            delay_s=float(config.value("actuators.brake.delay_s")),
            time_constant_s=float(
                config.value("actuators.brake.first_order_response_time_s")
            ),
        )
        self.shift_time = float(config.value("actuators.gear_shift.shift_time_s"))
        self.gear = 1
        self._shift_remaining = 0.0
        self.last_saturated = False

    def reset(
        self,
        *,
        steering: float = 0.0,
        throttle: float = 0.0,
        brake: float = 0.0,
        gear: int = 1,
    ) -> None:
        self.steering.reset(steering)
        self.throttle.reset(throttle)
        self.brake.reset(brake)
        self.gear = max(1, min(int(gear), 6))
        self._shift_remaining = 0.0
        self.last_saturated = False

    def update(self, command: VehicleCommand, dt: float) -> AppliedActuators:
        target_steer = clamp(command.steering_target, -self.max_steer, self.max_steer)
        target_throttle = clamp(command.throttle_target, 0.0, 1.0)
        target_brake = clamp(command.brake_target, 0.0, 1.0)
        self.last_saturated = (
            not math.isclose(target_steer, command.steering_target)
            or not math.isclose(target_throttle, command.throttle_target)
            or not math.isclose(target_brake, command.brake_target)
        )
        if command.emergency_brake:
            target_throttle = 0.0
            target_brake = 1.0
        elif target_throttle > 0.0 and target_brake > 0.0:
            self.last_saturated = True
            if target_brake >= target_throttle:
                target_throttle = 0.0
            else:
                target_brake = 0.0

        steering = self.steering.update(target_steer, dt)
        throttle = self.throttle.update(target_throttle, dt)
        brake = self.brake.update(target_brake, dt)
        if throttle > 0.02 and brake > 0.02:
            self.last_saturated = True
            if brake >= throttle:
                throttle = self.throttle.value = 0.0
            else:
                brake = self.brake.value = 0.0

        self._shift_remaining = max(0.0, self._shift_remaining - dt)
        requested_gear = max(1, min(int(command.gear_request), 6))
        if requested_gear != self.gear and self._shift_remaining <= 0.0:
            self.gear = requested_gear
            self._shift_remaining = self.shift_time
        if self._shift_remaining > 0.0:
            throttle = 0.0
        return AppliedActuators(
            steering=steering,
            throttle=throttle,
            brake=brake,
            gear=self.gear,
            saturated=self.last_saturated,
        )

