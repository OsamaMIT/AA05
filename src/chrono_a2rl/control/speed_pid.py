"""Conservative longitudinal speed PID controller."""

from __future__ import annotations

from typing import Any

from chrono_a2rl.common.math_utils import clamp
from chrono_a2rl.common.types import ControllerReference, TrackState, VehicleCommand, VehicleState
from chrono_a2rl.control.controller_interface import Controller


class SpeedPIDController(Controller):
    """PID controller that maps target speed to throttle/brake targets."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.kp_throttle = float(cfg.get("kp_throttle", 0.28))
        self.ki_throttle = float(cfg.get("ki_throttle", 0.03))
        self.kd_throttle = float(cfg.get("kd_throttle", 0.01))
        self.kp_brake = float(cfg.get("kp_brake", 0.35))
        self.ki_brake = float(cfg.get("ki_brake", 0.02))
        self.kd_brake = float(cfg.get("kd_brake", 0.01))
        self.integral_limit = float(cfg.get("integral_limit", 8.0))
        self.deadband = float(cfg.get("target_speed_deadband", 0.25))
        self._integral = 0.0
        self._previous_error = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = 0.0

    def compute_command(
        self,
        state: VehicleState,
        track_state: TrackState,
        reference: ControllerReference,
        dt: float,
    ) -> VehicleCommand:
        del track_state
        error = reference.target_speed - state.speed
        if abs(error) < self.deadband:
            error = 0.0
        dt_safe = max(dt, 1.0e-6)
        self._integral = clamp(
            self._integral + error * dt_safe,
            -self.integral_limit,
            self.integral_limit,
        )
        derivative = (error - self._previous_error) / dt_safe
        self._previous_error = error

        if error >= 0.0:
            throttle = (
                self.kp_throttle * error
                + self.ki_throttle * self._integral
                + self.kd_throttle * derivative
            )
            return VehicleCommand(
                throttle_target=clamp(throttle, 0.0, 1.0),
                brake_target=0.0,
                gear_request=max(1, state.gear),
                command_timestamp=state.sim_time,
                command_valid_until=state.sim_time + dt,
            )

        brake_error = -error
        brake = (
            self.kp_brake * brake_error
            - self.ki_brake * self._integral
            - self.kd_brake * derivative
        )
        return VehicleCommand(
            throttle_target=0.0,
            brake_target=clamp(brake, 0.0, 1.0),
            gear_request=max(1, state.gear),
            command_timestamp=state.sim_time,
            command_valid_until=state.sim_time + dt,
        )
