"""AVLite WorldBridge backed by direct Chrono/A2RL vehicle dynamics."""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

from avlite.c10_perception.c11_perception_model import EgoState, Map, PerceptionModel
from avlite.c30_control.c31_control_model import (
    AckermannControlCommand,
    ControlCommandBase,
)
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c50_common.c51_capabilities import (
    StackCapability,
    WorldCapability,
)
from avlite.c50_common.c52_world_sensor_datatypes import (
    ImuReading,
    SensorFrame,
    WheelOdometry,
)

from chrono_a2rl.avlite_stack.commands import A2RLAckermannCommand
from chrono_a2rl.avlite_stack.conversions import update_ego_from_vehicle
from chrono_a2rl.avlite_stack.settings import A2RLStackSettings
from chrono_a2rl.chrono_interface.direct_backend import ChronoDirectBackend
from chrono_a2rl.chrono_interface.reset_manager import initial_state_from_track
from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.common.types import VehicleCommand, VehicleState
from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_loader import load_track_from_config
from chrono_a2rl.vehicle.telemetry import VehicleTelemetry


class A2RLChronoWorldBridge(WorldBridge):
    """Canonical AVLite boundary around synchronous direct Chrono stepping."""

    world_capabilities = frozenset(
        {
            WorldCapability.IMU,
            WorldCapability.WHEEL_ENCODER,
        }
    )
    stack_capabilities = frozenset(
        {
            StackCapability.DETECTION,
            StackCapability.TRACKING,
            StackCapability.LOCALIZATION,
        }
    )
    stack_requirements = frozenset({StackCapability.CONTROL})

    def __init__(
        self,
        ego_state: EgoState,
        pm: PerceptionModel | None = None,
        reference_point: tuple[float, float] | None = None,
        map: Map | None = None,
        *,
        setting: type[A2RLStackSettings] = A2RLStackSettings,
        resolved_config: dict[str, Any] | None = None,
    ) -> None:
        self.setting = setting
        self.config = (
            resolved_config
            if resolved_config is not None
            else load_experiment_config(setting.experiment_config)
        )
        self.track = load_track_from_config(self.config["track"])
        self.speed_profile = generate_speed_profile(
            self.track, self.config.get("speed_profile", {})
        )
        self.vehicle_config = self.config["vehicle"]
        self.simulation_config = dict(self.config["simulation"])
        if resolved_config is None:
            if str(setting.backend).lower() != "configured":
                self.simulation_config["backend"] = str(setting.backend).lower()
            if str(setting.chrono_mode).lower() != "configured":
                self.simulation_config["chrono_mode"] = str(setting.chrono_mode).lower()

        self.ego_state = ego_state
        self.pm = pm or PerceptionModel(ego_vehicle=ego_state)
        self.pm.ego_vehicle = ego_state
        self.perception_model = self.pm
        self.reference_point = reference_point
        self.map = map
        self.backend = ChronoDirectBackend(
            self.vehicle_config, self.simulation_config
        )
        start_s = (
            float(setting.start_s_m)
            if resolved_config is None
            else float(self.config.get("avlite", {}).get("start_s_m", 0.0))
        )
        initial_speed = (
            float(setting.initial_speed_mps)
            if resolved_config is None
            else float(self.simulation_config.get("initial_speed", 12.0))
        )
        path_source = str(
            self.config.get("rl", {}).get("lateral_offset_reference", "raceline")
        ).lower()
        lateral_offset = (
            self.track.raceline_lateral_offset_at(start_s)
            if path_source == "raceline"
            else 0.0
        )
        self._initial_state = initial_state_from_track(
            self.track,
            s=start_s,
            speed=initial_speed,
            lateral_offset=lateral_offset,
            heading_source=path_source,
        )
        self.state = self.backend.reset(self._initial_state)
        self.last_vehicle_command = VehicleCommand()
        self.last_track_state = self.track.track_state_at_pose(
            self.state.x, self.state.y, self.state.yaw
        )
        self._last_command_wall_time = time.monotonic()
        update_ego_from_vehicle(self.ego_state, self.state)

    def control_ego_state(
        self, cmd: ControlCommandBase, dt: float | None = 0.01
    ) -> None:
        self.apply_vehicle_command(self._to_vehicle_command(cmd), float(dt or 0.01))

    def apply_vehicle_command(
        self, command: VehicleCommand, dt: float
    ) -> VehicleState:
        self.last_vehicle_command = replace(command)
        self._last_command_wall_time = time.monotonic()
        self.state = self.backend.step(command, dt)
        update_ego_from_vehicle(self.ego_state, self.state)
        self.last_track_state = self.track.track_state_at_pose(
            self.state.x, self.state.y, self.state.yaw
        )
        return replace(self.state)

    def step(self, dt: float | None = 0.01) -> None:
        """Advance under the held command for executers that support substeps."""

        timeout = float(self.setting.control_command_timeout_s)
        command = self.last_vehicle_command
        if timeout > 0.0 and time.monotonic() - self._last_command_wall_time > timeout:
            command = replace(command, throttle_target=0.0, brake_target=0.0)
        self.apply_vehicle_command(command, float(dt or 0.01))

    def get_ego_state(self) -> EgoState:
        return self.ego_state

    def get_ground_truth_perception_model(self) -> PerceptionModel:
        return self.pm

    def get_imu(self, agent_id: int = 0) -> ImuReading | None:
        self._require_ego_agent(agent_id, "imu")
        telemetry = self.get_vehicle_telemetry()
        return ImuReading(
            linear_accel=(
                telemetry.longitudinal_accel_mps2,
                telemetry.lateral_accel_mps2,
                0.0,
            ),
            angular_velocity=(0.0, 0.0, self.state.yaw_rate),
        )

    def get_wheel_odometry(self, agent_id: int = 0) -> WheelOdometry | None:
        self._require_ego_agent(agent_id, "wheel odometry")
        return WheelOdometry(
            linear_velocity=self.state.speed,
            yaw_rate=self.state.yaw_rate,
        )

    def get_sensor_frame(self, agent_id: int = 0) -> SensorFrame:
        self._require_ego_agent(agent_id, "sensor frame")
        return self._apply_world_capability_filter(
            SensorFrame(
                imu=self.get_imu(),
                wheel_odometry=self.get_wheel_odometry(),
                stamp=self.state.sim_time,
                frame_id="map",
            )
        )

    def teleport_ego(
        self, x: float, y: float, theta: float | None = None
    ) -> None:
        state = replace(
            self.state,
            x=float(x),
            y=float(y),
            yaw=self.state.yaw if theta is None else float(theta),
        )
        self.reset_vehicle(state)

    def reset_vehicle(self, initial_state: VehicleState | None = None) -> VehicleState:
        self.state = self.backend.reset(initial_state or self._initial_state)
        self.last_vehicle_command = VehicleCommand()
        self._last_command_wall_time = time.monotonic()
        update_ego_from_vehicle(self.ego_state, self.state)
        self.last_track_state = self.track.track_state_at_pose(
            self.state.x, self.state.y, self.state.yaw
        )
        return replace(self.state)

    def reset(self) -> None:
        self.pm.reset()
        self.pm.ego_vehicle = self.ego_state
        self.reset_vehicle()

    def close(self) -> None:
        self.backend.close()

    def get_vehicle_state(self) -> VehicleState:
        return replace(self.state)

    def get_vehicle_telemetry(self) -> VehicleTelemetry:
        telemetry = self.backend.get_telemetry()
        telemetry.off_track = not self.last_track_state.on_track
        telemetry.on_curb = self.last_track_state.on_curb
        return telemetry

    @property
    def last_control_saturated(self) -> bool:
        return self.backend.last_control_saturated

    def _to_vehicle_command(self, cmd: ControlCommandBase) -> VehicleCommand:
        if not isinstance(cmd, AckermannControlCommand):
            raise TypeError(
                "A2RLChronoWorldBridge requires an AckermannControlCommand"
            )
        max_accel = max(float(self.vehicle_config.get("max_accel", 1.0)), 1.0e-6)
        max_decel = max(float(self.vehicle_config.get("max_decel", 1.0)), 1.0e-6)
        if isinstance(cmd, A2RLAckermannCommand):
            throttle = float(cmd.throttle)
            brake = float(cmd.brake)
            gear = int(cmd.gear)
            emergency = bool(cmd.emergency_brake)
        else:
            acceleration = float(cmd.acceleration)
            throttle = max(acceleration, 0.0) / max_accel
            brake = max(-acceleration, 0.0) / max_decel
            gear = max(self.state.gear, 1)
            emergency = False
        return VehicleCommand(
            steering_target=float(cmd.steer),
            throttle_target=throttle,
            brake_target=brake,
            gear_request=gear,
            emergency_brake=emergency,
            command_timestamp=self.state.sim_time,
            command_valid_until=self.state.sim_time
            + float(self.simulation_config.get("control_dt", 0.02)),
        )
