"""Direct Chrono backend abstraction with a runnable mock fallback."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from chrono_a2rl.chrono_interface.dbw_model import DBWModel
from chrono_a2rl.common.logging import get_logger
from chrono_a2rl.common.math_utils import clamp, wrap_angle
from chrono_a2rl.common.types import VehicleCommand, VehicleState
from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig
from chrono_a2rl.vehicle.dynamic_bicycle import DynamicBicycleModel
from chrono_a2rl.vehicle.kinematic_bicycle import KinematicBicycleModel
from chrono_a2rl.vehicle.telemetry import VehicleTelemetry


LOGGER = get_logger(__name__)


def _import_pychrono():
    try:
        import pychrono as chrono  # type: ignore
    except ImportError:
        return None
    return chrono


def _chrono_available() -> bool:
    return _import_pychrono() is not None


class MockChronoBackend:
    """Simple kinematic bicycle simulator used when Chrono is unavailable."""

    def __init__(
        self,
        vehicle_config: dict[str, Any] | None = None,
        simulation_config: dict[str, Any] | None = None,
    ) -> None:
        self.vehicle_config = vehicle_config or {}
        self.simulation_config = simulation_config or {}
        self.wheelbase = float(self.vehicle_config.get("wheelbase", 2.97))
        self.max_accel = float(self.vehicle_config.get("max_accel", 4.2))
        self.max_decel = float(self.vehicle_config.get("max_decel", 9.0))
        self.max_speed = float(self.vehicle_config.get("max_speed", 55.0))
        self.drag_coefficient = float(self.vehicle_config.get("drag_coefficient", 0.018))
        self.rolling_resistance = float(self.vehicle_config.get("rolling_resistance", 0.08))
        self.dbw = DBWModel(self.vehicle_config)
        self.state = VehicleState()

    def reset(self, initial_state: VehicleState | None = None) -> VehicleState:
        """Reset simulation state."""

        self.dbw.reset()
        self.state = replace(initial_state) if initial_state is not None else VehicleState()
        return replace(self.state)

    def step(self, command: VehicleCommand, dt: float) -> VehicleState:
        """Advance the mock vehicle by one control step."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        applied = self.dbw.apply(command, self.state, dt)
        speed = max(0.0, self.state.speed)
        drag = self.drag_coefficient * speed * speed
        rolling = self.rolling_resistance if speed > 0.1 else 0.0
        accel = (
            applied.throttle_target * self.max_accel
            - applied.brake_target * self.max_decel
            - drag
            - rolling
        )
        new_speed = clamp(speed + accel * dt, 0.0, self.max_speed)
        yaw_rate = 0.0
        if self.wheelbase > 0.0:
            yaw_rate = new_speed / self.wheelbase * math.tan(applied.steering_target)
        yaw = wrap_angle(self.state.yaw + yaw_rate * dt)
        x = self.state.x + new_speed * math.cos(yaw) * dt
        y = self.state.y + new_speed * math.sin(yaw) * dt
        vx = new_speed * math.cos(yaw)
        vy = new_speed * math.sin(yaw)
        self.state = VehicleState(
            x=x,
            y=y,
            z=self.state.z,
            yaw=yaw,
            pitch=self.state.pitch,
            roll=self.state.roll,
            vx=vx,
            vy=vy,
            vz=0.0,
            speed=new_speed,
            yaw_rate=yaw_rate,
            steering_angle=applied.steering_target,
            throttle=applied.throttle_target,
            brake=applied.brake_target,
            gear=applied.gear_request,
            sim_time=self.state.sim_time + dt,
        )
        return replace(self.state)

    def get_state(self) -> VehicleState:
        """Return current state."""

        return replace(self.state)

    def get_telemetry(self) -> VehicleTelemetry:
        """Return compatibility telemetry for the Level 0 comparison model."""

        return VehicleTelemetry(
            speed_mps=self.state.speed,
            speed_kmh=self.state.speed * 3.6,
            yaw_rate_rad_s=self.state.yaw_rate,
            steering_actual_rad=self.state.steering_angle,
            throttle_actual=self.state.throttle,
            brake_actual=self.state.brake,
            gear=self.state.gear,
        )

    @property
    def last_control_saturated(self) -> bool:
        """Return whether the DBW model saturated the most recent command."""

        return self.dbw.last_saturated

    def close(self) -> None:
        """Close backend resources."""

        return None


class PyChronoKinematicBackend(MockChronoBackend):
    """PyChrono-backed kinematic backend.

    This is the first real Chrono mode for the repository. It creates a Chrono
    system, a flat ground body, and a chassis body, then mirrors the scaffold's
    kinematic A2RL-style state into that Chrono world each control step. The
    public backend API is identical to the mock backend, so this can later be
    replaced internally by a full `pychrono.vehicle` model without changing the
    controller, track, logging, or RL code.
    """

    def __init__(
        self,
        vehicle_config: dict[str, Any] | None = None,
        simulation_config: dict[str, Any] | None = None,
    ) -> None:
        self.chrono = _import_pychrono()
        if self.chrono is None:
            raise ImportError("pychrono is not installed")
        super().__init__(vehicle_config, simulation_config)
        self.physics_dt = float(self.simulation_config.get("physics_dt", 0.001))
        self.system = self._make_system()
        self.chassis = self._make_chassis_body()
        self.ground = self._make_ground_body()
        self._add_body(self.ground)
        self._add_body(self.chassis)
        LOGGER.info("Using PyChrono kinematic backend.")

    def reset(self, initial_state: VehicleState | None = None) -> VehicleState:
        state = super().reset(initial_state)
        self._sync_chassis_from_state(state)
        return state

    def step(self, command: VehicleCommand, dt: float) -> VehicleState:
        state = super().step(command, dt)
        self._sync_chassis_from_state(state)
        self._step_chrono(dt)
        return state

    def close(self) -> None:
        return None

    def _make_system(self):
        chrono = self.chrono
        system_cls = getattr(chrono, "ChSystemNSC", None) or getattr(chrono, "ChSystem", None)
        if system_cls is None:
            raise RuntimeError("Could not find ChSystemNSC or ChSystem in pychrono")
        system = system_cls()
        gravity = self._vec3(0.0, 0.0, -9.81)
        if hasattr(system, "SetGravitationalAcceleration"):
            system.SetGravitationalAcceleration(gravity)
        elif hasattr(system, "Set_G_acc"):
            system.Set_G_acc(gravity)
        return system

    def _make_chassis_body(self):
        chrono = self.chrono
        body = chrono.ChBody()
        mass = float(self.vehicle_config.get("mass", 795.0))
        length = float(self.vehicle_config.get("body_length", 5.23))
        width = float(self.vehicle_config.get("body_width", 1.91))
        height = float(self.vehicle_config.get("body_height", 0.96))
        ixx = mass * (width * width + height * height) / 12.0
        iyy = mass * (length * length + height * height) / 12.0
        izz = mass * (length * length + width * width) / 12.0
        body.SetMass(mass)
        body.SetInertiaXX(self._vec3(ixx, iyy, izz))
        if hasattr(body, "SetFixed"):
            body.SetFixed(False)
        elif hasattr(body, "SetBodyFixed"):
            body.SetBodyFixed(False)
        self._add_visual_box(body, length, width, height)
        return body

    def _make_ground_body(self):
        chrono = self.chrono
        body = chrono.ChBody()
        body.SetMass(1.0)
        if hasattr(body, "SetFixed"):
            body.SetFixed(True)
        elif hasattr(body, "SetBodyFixed"):
            body.SetBodyFixed(True)
        body.SetPos(self._vec3(0.0, 0.0, -0.05))
        return body

    def _add_body(self, body) -> None:
        if hasattr(self.system, "AddBody"):
            self.system.AddBody(body)
        else:
            self.system.Add(body)

    def _sync_chassis_from_state(self, state: VehicleState) -> None:
        self.chassis.SetPos(self._vec3(state.x, state.y, state.z))
        self.chassis.SetRot(self._quat_from_yaw(state.yaw))
        velocity = self._vec3(state.vx, state.vy, state.vz)
        angular_velocity = self._vec3(0.0, 0.0, state.yaw_rate)
        if hasattr(self.chassis, "SetPosDt"):
            self.chassis.SetPosDt(velocity)
        if hasattr(self.chassis, "SetAngVelParent"):
            self.chassis.SetAngVelParent(angular_velocity)

    def _add_visual_box(self, body, length: float, width: float, height: float) -> None:
        chrono = self.chrono
        shape_cls = getattr(chrono, "ChVisualShapeBox", None)
        if shape_cls is None or not hasattr(body, "AddVisualShape"):
            return
        try:
            shape = shape_cls(length, width, height)
        except TypeError:
            shape = shape_cls()
            if hasattr(shape, "SetSize"):
                shape.SetSize(self._vec3(length, width, height))
        body.AddVisualShape(shape)

    def _step_chrono(self, dt: float) -> None:
        remaining = max(0.0, dt)
        step = self.physics_dt if self.physics_dt > 0.0 else dt
        while remaining > 1.0e-12:
            substep = min(step, remaining)
            if hasattr(self.system, "DoStepDynamics"):
                self.system.DoStepDynamics(substep)
            remaining -= substep

    def _vec3(self, x: float, y: float, z: float):
        chrono = self.chrono
        vector_cls = getattr(chrono, "ChVector3d", None) or getattr(chrono, "ChVectorD", None)
        if vector_cls is None:
            raise RuntimeError("Could not find ChVector3d or ChVectorD in pychrono")
        return vector_cls(float(x), float(y), float(z))

    def _quat_from_yaw(self, yaw: float):
        chrono = self.chrono
        if hasattr(chrono, "QuatFromAngleZ"):
            return chrono.QuatFromAngleZ(float(yaw))
        quat_cls = getattr(chrono, "ChQuaterniond", None) or getattr(chrono, "ChQuaternionD", None)
        if quat_cls is None:
            raise RuntimeError("Could not find ChQuaterniond or ChQuaternionD in pychrono")
        return quat_cls(math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


class A2RLDynamicsBackend:
    """Backend adapter for Level 1 and Level 2 non-Chrono vehicle models."""

    def __init__(
        self,
        vehicle_config: dict[str, Any],
        simulation_config: dict[str, Any],
        *,
        level: int = 2,
    ) -> None:
        self.config = A2RLVehicleConfig.from_component_config(vehicle_config)
        if level == 1:
            self.model = KinematicBicycleModel(self.config)
            LOGGER.info("Using A2RL Level 1 constrained kinematic bicycle.")
        else:
            self.model = DynamicBicycleModel(
                self.config,
                physics_dt=float(simulation_config.get("physics_dt", 0.001)),
            )
            LOGGER.info("Using A2RL Level 2 dynamic bicycle backend.")

    def reset(self, initial_state: VehicleState | None = None) -> VehicleState:
        return self.model.reset(initial_state)

    def step(self, command: VehicleCommand, dt: float) -> VehicleState:
        return self.model.step(command, dt)

    def get_state(self) -> VehicleState:
        return self.model.get_state()

    def get_telemetry(self) -> VehicleTelemetry:
        return self.model.get_telemetry()

    @property
    def last_control_saturated(self) -> bool:
        return self.model.last_control_saturated

    def close(self) -> None:
        return None


class PyChronoA2RLForceBackend(A2RLDynamicsBackend):
    """Level 3 Chrono rigid body driven by the shared force-limited model.

    Tire, powertrain, aero, and actuator forces are resolved by the tested
    Level 2 model. The resulting rigid-body state is mirrored into Chrono.
    This deliberately stops short of claiming a private multibody suspension.
    """

    def __init__(
        self,
        vehicle_config: dict[str, Any],
        simulation_config: dict[str, Any],
    ) -> None:
        chrono = _import_pychrono()
        if chrono is None:
            raise ImportError("pychrono is not installed")
        super().__init__(vehicle_config, simulation_config, level=2)
        from chrono_a2rl.chrono_interface.chrono_vehicle_factory import (
            create_a2rl_chrono_body,
        )

        self.chrono = chrono
        self.system, self.chrono_body = create_a2rl_chrono_body(chrono, self.config)
        self.physics_dt = float(simulation_config.get("physics_dt", 0.001))
        LOGGER.info("Using A2RL Level 3 PyChrono force-body backend.")

    def reset(self, initial_state: VehicleState | None = None) -> VehicleState:
        state = super().reset(initial_state)
        self.chrono_body.sync(state)
        return state

    def step(self, command: VehicleCommand, dt: float) -> VehicleState:
        state = super().step(command, dt)
        self.chrono_body.sync(state)
        return state


class ChronoDirectBackend:
    """Stable backend facade for Project Chrono or mock simulation."""

    def __init__(
        self,
        vehicle_config: dict[str, Any] | None = None,
        simulation_config: dict[str, Any] | None = None,
    ) -> None:
        self.vehicle_config = vehicle_config or {}
        self.simulation_config = simulation_config or {}
        requested_backend = str(self.simulation_config.get("backend", "mock")).lower()
        level = int(self.vehicle_config.get("dynamics_level", 0))
        mode = str(
            self.simulation_config.get(
                "chrono_mode",
                self.vehicle_config.get("default_model", "kinematic"),
            )
        ).lower()
        if "model_root" in self.vehicle_config and mode in {
            "dynamic_bicycle",
            "a2rl_dynamic",
            "level2",
        }:
            self._backend = A2RLDynamicsBackend(
                self.vehicle_config,
                self.simulation_config,
                level=max(level, 2),
            )
            return
        if "model_root" in self.vehicle_config and mode in {
            "constrained_kinematic",
            "level1",
        }:
            self._backend = A2RLDynamicsBackend(
                self.vehicle_config,
                self.simulation_config,
                level=1,
            )
            return
        if (
            requested_backend == "chrono"
            and "model_root" in self.vehicle_config
            and mode in {"a2rl_force_body", "dynamic_force", "level3"}
        ):
            try:
                self._backend = PyChronoA2RLForceBackend(
                    self.vehicle_config, self.simulation_config
                )
                return
            except Exception as exc:
                LOGGER.warning(
                    "A2RL PyChrono force-body initialization failed (%s). "
                    "Falling back to the Level 2 dynamic bicycle.",
                    exc,
                )
                self._backend = A2RLDynamicsBackend(
                    self.vehicle_config, self.simulation_config, level=2
                )
                return
        if requested_backend == "chrono":
            try:
                self._backend = PyChronoKinematicBackend(self.vehicle_config, self.simulation_config)
                return
            except Exception as exc:
                LOGGER.warning(
                    "Project Chrono backend could not be initialized (%s). "
                    "Falling back to MockChronoBackend.",
                    exc,
                )
        self._backend = MockChronoBackend(self.vehicle_config, self.simulation_config)

    def reset(self, initial_state: VehicleState | None = None) -> VehicleState:
        return self._backend.reset(initial_state)

    def step(self, command: VehicleCommand, dt: float) -> VehicleState:
        return self._backend.step(command, dt)

    def get_state(self) -> VehicleState:
        return self._backend.get_state()

    def get_telemetry(self) -> VehicleTelemetry:
        return self._backend.get_telemetry()

    @property
    def last_control_saturated(self) -> bool:
        return self._backend.last_control_saturated

    def close(self) -> None:
        self._backend.close()
