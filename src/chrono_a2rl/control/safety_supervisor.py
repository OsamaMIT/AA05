"""Runtime safety supervisor for DBW commands."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from chrono_a2rl.common.math_utils import clamp, has_nan
from chrono_a2rl.common.types import TrackState, VehicleCommand, VehicleState


class SafetySupervisor:
    """Clamp or override commands when basic safety checks fail."""

    def __init__(
        self,
        vehicle_config: dict[str, Any] | None = None,
        simulation_config: dict[str, Any] | None = None,
    ) -> None:
        vehicle = vehicle_config or {}
        sim = simulation_config or {}
        self.max_steer = float(vehicle.get("max_steer", 0.45))
        self.max_throttle = float(vehicle.get("max_throttle", 1.0))
        self.max_brake = float(vehicle.get("max_brake", 1.0))
        self.max_speed = float(vehicle.get("max_speed", 55.0))
        self.max_abs_yaw_rate = float(sim.get("max_abs_yaw_rate", 2.5))
        self.last_reason = "ok"
        self.last_saturated = False

    def supervise(
        self,
        command: VehicleCommand,
        state: VehicleState,
        track_state: TrackState,
        dt: float,
    ) -> VehicleCommand:
        """Return a safe command."""

        del dt
        self.last_reason = "ok"
        self.last_saturated = False

        command_values = (
            command.steering_target,
            command.throttle_target,
            command.brake_target,
            command.command_timestamp,
            command.command_valid_until,
        )
        if has_nan(*command_values):
            self.last_reason = "nan_command"
            return self._emergency_command(state)

        if not track_state.on_track:
            self.last_reason = "off_track"
            return self._emergency_command(state)

        if abs(state.yaw_rate) > self.max_abs_yaw_rate:
            self.last_reason = "excessive_yaw_rate"
            return self._emergency_command(state)

        if state.speed > self.max_speed:
            self.last_reason = "excessive_speed"
            return replace(
                command,
                throttle_target=0.0,
                brake_target=self.max_brake,
                emergency_brake=True,
            )

        steering = clamp(command.steering_target, -self.max_steer, self.max_steer)
        throttle = clamp(command.throttle_target, 0.0, self.max_throttle)
        brake = clamp(command.brake_target, 0.0, self.max_brake)
        self.last_saturated = (
            steering != command.steering_target
            or throttle != command.throttle_target
            or brake != command.brake_target
        )

        if throttle > 0.0 and brake > 0.0:
            self.last_reason = "throttle_brake_conflict"
            self.last_saturated = True
            if brake >= throttle:
                throttle = 0.0
            else:
                brake = 0.0

        return replace(
            command,
            steering_target=steering,
            throttle_target=throttle,
            brake_target=brake,
        )

    def _emergency_command(self, state: VehicleState) -> VehicleCommand:
        self.last_saturated = True
        return VehicleCommand(
            steering_target=clamp(state.steering_angle, -self.max_steer, self.max_steer),
            throttle_target=0.0,
            brake_target=self.max_brake,
            gear_request=max(1, state.gear),
            emergency_brake=True,
            command_timestamp=state.sim_time,
            command_valid_until=state.sim_time + 0.1,
        )
