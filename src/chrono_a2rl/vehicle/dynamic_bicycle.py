"""Force-limited planar dynamic bicycle model for A2RL research."""

from __future__ import annotations

from dataclasses import replace
import math

from chrono_a2rl.common.math_utils import clamp, wrap_angle
from chrono_a2rl.common.types import VehicleCommand, VehicleState
from chrono_a2rl.vehicle.actuator_model import A2RLActuatorModel
from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig
from chrono_a2rl.vehicle.aero_models import QuadraticAeroModel
from chrono_a2rl.vehicle.brake_model import A2RLBrakeModel
from chrono_a2rl.vehicle.powertrain_model import A2RLPowertrainModel
from chrono_a2rl.vehicle.suspension_load_transfer import SuspensionLoadTransferModel
from chrono_a2rl.vehicle.telemetry import VehicleTelemetry
from chrono_a2rl.vehicle.tire_models import TireForceResult, make_axle_tire_models
from chrono_a2rl.vehicle.vehicle_dynamics_base import VehicleDynamicsModel


class DynamicBicycleModel(VehicleDynamicsModel):
    """Level 2 model with axle slip, combined grip, aero, and DBW dynamics."""

    def __init__(
        self,
        config: A2RLVehicleConfig,
        *,
        physics_dt: float = 0.001,
    ) -> None:
        self.config = config
        self.physics_dt = max(float(physics_dt), 1.0e-4)
        self.mass = float(config.value("vehicle.mass.total_kg"))
        self.yaw_inertia = float(config.value("vehicle.inertia.iz_kgm2"))
        self.lf = float(config.value("geometry.cg_to_front_axle_m"))
        self.lr = float(config.value("geometry.cg_to_rear_axle_m"))
        self.max_speed = float(
            config.value("powertrain.longitudinal_limits.max_speed_mps")
        )
        self.front_mu = float(config.value("tires.friction.mu_peak_front"))
        self.rear_mu = float(config.value("tires.friction.mu_peak_rear"))
        self.aero = QuadraticAeroModel(config)
        self.load_transfer = SuspensionLoadTransferModel(config)
        self.front_tire, self.rear_tire = make_axle_tire_models(config)
        self.brakes = A2RLBrakeModel(config)
        self.powertrain = A2RLPowertrainModel(config)
        self.actuators = A2RLActuatorModel(config)
        self.state = VehicleState()
        self.telemetry = VehicleTelemetry()
        self._vx_body = 0.0
        self._vy_body = 0.0
        self._ax = 0.0
        self._ay = 0.0
        self._last_command = VehicleCommand()

    def reset(self, initial_state: VehicleState | None = None) -> VehicleState:
        self.state = replace(initial_state) if initial_state is not None else VehicleState()
        if self.state.speed > 0.0:
            self._vx_body = self.state.speed
            self._vy_body = 0.0
        else:
            cos_yaw = math.cos(self.state.yaw)
            sin_yaw = math.sin(self.state.yaw)
            self._vx_body = self.state.vx * cos_yaw + self.state.vy * sin_yaw
            self._vy_body = -self.state.vx * sin_yaw + self.state.vy * cos_yaw
        self._vx_body = max(self._vx_body, 0.0)
        self._ax = 0.0
        self._ay = 0.0
        self.actuators.reset(
            steering=self.state.steering_angle,
            throttle=self.state.throttle,
            brake=self.state.brake,
            gear=self.state.gear,
        )
        self.telemetry = VehicleTelemetry(
            speed_mps=self.state.speed,
            speed_kmh=self.state.speed * 3.6,
            yaw_rate_rad_s=self.state.yaw_rate,
            steering_actual_rad=self.state.steering_angle,
            throttle_actual=self.state.throttle,
            brake_actual=self.state.brake,
            gear=self.state.gear,
        )
        return replace(self.state)

    def step(self, command: VehicleCommand, dt: float) -> VehicleState:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self._last_command = replace(command)
        remaining = float(dt)
        while remaining > 1.0e-12:
            substep = min(self.physics_dt, remaining)
            self._integrate_substep(command, substep)
            remaining -= substep
        return replace(self.state)

    def _integrate_substep(self, command: VehicleCommand, dt: float) -> None:
        applied = self.actuators.update(command, dt)
        vx = max(self._vx_body, 0.0)
        vy = self._vy_body
        yaw_rate = self.state.yaw_rate
        aero = self.aero.forces(vx)
        loads = self.load_transfer.loads(
            longitudinal_accel_mps2=self._ax,
            lateral_accel_mps2=self._ay,
            aero=aero,
        )
        brake_request = self.brakes.force_request(applied.brake)
        power = self.powertrain.output(
            speed_mps=vx,
            throttle=applied.throttle,
            rear_normal_load_n=loads.rear_normal_load_n,
            rear_mu=self.rear_mu,
            gear=applied.gear,
        )

        slip_speed = max(vx, 3.0)
        alpha_front = applied.steering - math.atan2(
            vy + self.lf * yaw_rate, slip_speed
        )
        alpha_rear = -math.atan2(vy - self.lr * yaw_rate, slip_speed)
        front = self.front_tire.forces(
            normal_load_n=loads.front_normal_load_n,
            slip_angle_rad=alpha_front,
            longitudinal_request_n=-brake_request.front_request_n,
        )
        rear = self.rear_tire.forces(
            normal_load_n=loads.rear_normal_load_n,
            slip_angle_rad=alpha_rear,
            longitudinal_request_n=(
                power.drive_force_n - brake_request.rear_request_n
            ),
        )

        sin_delta = math.sin(applied.steering)
        cos_delta = math.cos(applied.steering)
        fx_body = (
            rear.longitudinal_force_n
            + front.longitudinal_force_n * cos_delta
            - front.lateral_force_n * sin_delta
            - aero.drag_force_n
            - power.rolling_resistance_n
        )
        fy_body = (
            rear.lateral_force_n
            + front.lateral_force_n * cos_delta
            + front.longitudinal_force_n * sin_delta
        )
        ax = fx_body / self.mass + vy * yaw_rate
        ay = fy_body / self.mass - vx * yaw_rate
        yaw_accel = (
            self.lf
            * (
                front.lateral_force_n * cos_delta
                + front.longitudinal_force_n * sin_delta
            )
            - self.lr * rear.lateral_force_n
        ) / self.yaw_inertia

        # Force saturation is physical; these derivative guards only protect
        # explicit integration from numerical blow-up at very low speed.
        ax = clamp(ax, -35.0, 20.0)
        ay = clamp(ay, -55.0, 55.0)
        yaw_accel = clamp(yaw_accel, -25.0, 25.0)
        new_vx = clamp(vx + ax * dt, 0.0, self.max_speed * 1.01)
        new_vy = vy + ay * dt
        if new_vx < 1.0:
            new_vy *= max(0.0, 1.0 - 8.0 * dt)
        new_yaw_rate = yaw_rate + yaw_accel * dt
        new_yaw = wrap_angle(self.state.yaw + new_yaw_rate * dt)
        world_vx = new_vx * math.cos(new_yaw) - new_vy * math.sin(new_yaw)
        world_vy = new_vx * math.sin(new_yaw) + new_vy * math.cos(new_yaw)
        new_x = self.state.x + world_vx * dt
        new_y = self.state.y + world_vy * dt
        speed = math.hypot(new_vx, new_vy)
        actual_brake = min(
            brake_request.total_request_n,
            max(-front.longitudinal_force_n, 0.0)
            + max(power.drive_force_n - rear.longitudinal_force_n, 0.0),
        )
        self._vx_body = new_vx
        self._vy_body = new_vy
        self._ax = ax
        self._ay = fy_body / self.mass
        self.state = VehicleState(
            x=new_x,
            y=new_y,
            z=self.state.z,
            yaw=new_yaw,
            pitch=self.state.pitch,
            roll=self.state.roll,
            vx=world_vx,
            vy=world_vy,
            vz=0.0,
            speed=speed,
            yaw_rate=new_yaw_rate,
            steering_angle=applied.steering,
            throttle=applied.throttle,
            brake=applied.brake,
            gear=power.gear,
            sim_time=self.state.sim_time + dt,
        )
        self.telemetry = self._telemetry_from_forces(
            command,
            applied.steering,
            applied.throttle,
            applied.brake,
            front,
            rear,
            loads.front_normal_load_n,
            loads.rear_normal_load_n,
            aero.drag_force_n,
            aero.downforce_total_n,
            power.drive_force_n,
            actual_brake,
        )

    def _telemetry_from_forces(
        self,
        command: VehicleCommand,
        steering: float,
        throttle: float,
        brake: float,
        front: TireForceResult,
        rear: TireForceResult,
        front_load: float,
        rear_load: float,
        drag: float,
        downforce: float,
        drive_force: float,
        brake_force: float,
    ) -> VehicleTelemetry:
        return VehicleTelemetry(
            speed_mps=self.state.speed,
            speed_kmh=self.state.speed * 3.6,
            longitudinal_accel_mps2=self._ax,
            lateral_accel_mps2=self._ay,
            yaw_rate_rad_s=self.state.yaw_rate,
            slip_angle_front_rad=front.slip_angle_rad,
            slip_angle_rear_rad=rear.slip_angle_rad,
            tire_usage_front=front.usage_ratio,
            tire_usage_rear=rear.usage_ratio,
            combined_slip_usage=max(front.usage_ratio, rear.usage_ratio),
            front_normal_load_n=front_load,
            rear_normal_load_n=rear_load,
            downforce_n=downforce,
            drag_force_n=drag,
            drive_force_n=drive_force,
            brake_force_n=brake_force,
            steering_target_rad=command.steering_target,
            steering_actual_rad=steering,
            throttle_target=command.throttle_target,
            throttle_actual=throttle,
            brake_target=command.brake_target,
            brake_actual=brake,
            gear=self.state.gear,
        )

    def get_state(self) -> VehicleState:
        return replace(self.state)

    def get_telemetry(self) -> VehicleTelemetry:
        return replace(self.telemetry)

    @property
    def last_control_saturated(self) -> bool:
        return self.actuators.last_saturated

