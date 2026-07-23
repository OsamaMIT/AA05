"""Level 1 kinematic bicycle with realistic actuator and aero limits."""

from __future__ import annotations

from dataclasses import replace
import math

from chrono_a2rl.common.math_utils import clamp, wrap_angle
from chrono_a2rl.common.types import VehicleCommand, VehicleState
from chrono_a2rl.vehicle.actuator_model import A2RLActuatorModel
from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig
from chrono_a2rl.vehicle.aero_models import QuadraticAeroModel
from chrono_a2rl.vehicle.telemetry import VehicleTelemetry
from chrono_a2rl.vehicle.vehicle_dynamics_base import VehicleDynamicsModel


class KinematicBicycleModel(VehicleDynamicsModel):
    """Comparison model with finite DBW response, power, and aero drag."""

    def __init__(self, config: A2RLVehicleConfig) -> None:
        self.config = config
        self.mass = float(config.value("vehicle.mass.total_kg"))
        self.wheelbase = float(config.value("geometry.wheelbase_m"))
        self.max_speed = float(
            config.value("powertrain.longitudinal_limits.max_speed_mps")
        )
        self.max_accel = float(
            config.value("powertrain.longitudinal_limits.max_accel_low_speed_mps2")
        )
        self.max_decel = float(config.value("brakes.max_decel_mps2"))
        self.aero = QuadraticAeroModel(config)
        self.actuators = A2RLActuatorModel(config)
        self.state = VehicleState()
        self.telemetry = VehicleTelemetry()

    def reset(self, initial_state: VehicleState | None = None) -> VehicleState:
        self.state = replace(initial_state) if initial_state else VehicleState()
        self.actuators.reset(
            steering=self.state.steering_angle,
            throttle=self.state.throttle,
            brake=self.state.brake,
            gear=self.state.gear,
        )
        return replace(self.state)

    def step(self, command: VehicleCommand, dt: float) -> VehicleState:
        applied = self.actuators.update(command, dt)
        aero = self.aero.forces(self.state.speed)
        accel = (
            applied.throttle * self.max_accel
            - applied.brake * self.max_decel
            - aero.drag_force_n / self.mass
        )
        speed = clamp(self.state.speed + accel * dt, 0.0, self.max_speed)
        yaw_rate = speed / self.wheelbase * math.tan(applied.steering)
        yaw = wrap_angle(self.state.yaw + yaw_rate * dt)
        vx = speed * math.cos(yaw)
        vy = speed * math.sin(yaw)
        self.state = VehicleState(
            x=self.state.x + vx * dt,
            y=self.state.y + vy * dt,
            z=self.state.z,
            yaw=yaw,
            vx=vx,
            vy=vy,
            speed=speed,
            yaw_rate=yaw_rate,
            steering_angle=applied.steering,
            throttle=applied.throttle,
            brake=applied.brake,
            gear=applied.gear,
            sim_time=self.state.sim_time + dt,
        )
        self.telemetry = VehicleTelemetry(
            speed_mps=speed,
            speed_kmh=speed * 3.6,
            longitudinal_accel_mps2=accel,
            yaw_rate_rad_s=yaw_rate,
            downforce_n=aero.downforce_total_n,
            drag_force_n=aero.drag_force_n,
            steering_target_rad=command.steering_target,
            steering_actual_rad=applied.steering,
            throttle_target=command.throttle_target,
            throttle_actual=applied.throttle,
            brake_target=command.brake_target,
            brake_actual=applied.brake,
            gear=applied.gear,
        )
        return replace(self.state)

    def get_state(self) -> VehicleState:
        return replace(self.state)

    def get_telemetry(self) -> VehicleTelemetry:
        return replace(self.telemetry)

    @property
    def last_control_saturated(self) -> bool:
        return self.actuators.last_saturated
