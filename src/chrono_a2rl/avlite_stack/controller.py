"""AVLite control strategy backed by the existing A2RL MPC/PID baseline."""

from __future__ import annotations

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame

from chrono_a2rl.avlite_stack.commands import A2RLAckermannCommand
from chrono_a2rl.avlite_stack.conversions import ego_to_vehicle_state
from chrono_a2rl.avlite_stack.settings import A2RLStackSettings
from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.common.types import VehicleCommand
from chrono_a2rl.control.mpc_lateral import LateralMPCController
from chrono_a2rl.control.reference_generator import make_reference
from chrono_a2rl.control.safety_supervisor import SafetySupervisor
from chrono_a2rl.control.speed_pid import SpeedPIDController
from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_loader import load_track_from_config


class A2RLMPCControlStrategy(ControlStrategy):
    """Track the TUMFTM raceline and speed profile through AVLite control."""

    def __init__(self, tj=None, setting: type[A2RLStackSettings] = A2RLStackSettings):
        super().__init__(tj=tj)
        self.setting = setting
        self.config = load_experiment_config(setting.experiment_config)
        self.track = load_track_from_config(self.config["track"])
        self.speed_profile = generate_speed_profile(
            self.track, self.config.get("speed_profile", {})
        )
        self.lateral = LateralMPCController(
            self.config["controller"].get("lateral", {}), self.config["vehicle"]
        )
        self.speed = SpeedPIDController(
            self.config["controller"].get("speed", {})
        )
        self.supervisor = SafetySupervisor(
            self.config["vehicle"], self.config["simulation"]
        )
        self.path_source = str(
            self.config.get("rl", {}).get("lateral_offset_reference", "raceline")
        ).lower()
        self._steering_feedback = 0.0
        self._throttle_feedback = 0.0
        self._brake_feedback = 0.0
        self._gear_feedback = 1
        self.last_reference = None
        self.last_track_state = None
        self.last_safe_command = VehicleCommand()

    def control(
        self,
        ego: EgoState,
        plan: GlobalPlan | LocalPlan | None = None,
        control_dt: float | None = None,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> A2RLAckermannCommand:
        del perception_model
        if plan is not None:
            self.set_plan(plan)
        dt = float(control_dt or self.config["simulation"].get("control_dt", 0.02))
        sim_time = float(sensors.stamp or 0.0) if sensors is not None else 0.0
        state = ego_to_vehicle_state(
            ego,
            sensors=sensors,
            steering_angle=self._steering_feedback,
            throttle=self._throttle_feedback,
            brake=self._brake_feedback,
            gear=self._gear_feedback,
            sim_time=sim_time,
        )
        track_state = self.track.track_state_at_pose(state.x, state.y, state.yaw)
        current_offset = (
            self.track.raceline_lateral_offset_at(track_state.s)
            if self.path_source == "raceline"
            else 0.0
        )
        preview_s = track_state.s + self.speed_profile.speed_at(track_state.s) * 0.25
        preview_offset = (
            self.track.raceline_lateral_offset_at(preview_s)
            if self.path_source == "raceline"
            else 0.0
        )
        reference = make_reference(
            self.track,
            self.speed_profile,
            track_state,
            lateral_offset=current_offset,
            lookahead_lateral_offset=preview_offset,
            path_source=self.path_source,
            horizon_steps=self.lateral.horizon_steps,
            control_dt=dt,
        )
        steering = self.lateral.compute_command(state, track_state, reference, dt)
        longitudinal = self.speed.compute_command(state, track_state, reference, dt)
        requested = VehicleCommand(
            steering_target=steering.steering_target,
            throttle_target=longitudinal.throttle_target,
            brake_target=longitudinal.brake_target,
            gear_request=longitudinal.gear_request,
            command_timestamp=sim_time,
            command_valid_until=sim_time + dt,
        )
        safe = self.supervisor.supervise(requested, state, track_state, dt)
        max_accel = max(float(self.config["vehicle"].get("max_accel", 1.0)), 1.0e-6)
        max_decel = max(float(self.config["vehicle"].get("max_decel", 1.0)), 1.0e-6)
        acceleration = (
            safe.throttle_target * max_accel - safe.brake_target * max_decel
        )
        self._steering_feedback = safe.steering_target
        self._throttle_feedback = safe.throttle_target
        self._brake_feedback = safe.brake_target
        self._gear_feedback = safe.gear_request
        self.last_reference = reference
        self.last_track_state = track_state
        self.last_safe_command = safe
        result = A2RLAckermannCommand(
            steer=safe.steering_target,
            acceleration=acceleration,
            throttle=safe.throttle_target,
            brake=safe.brake_target,
            gear=safe.gear_request,
            emergency_brake=safe.emergency_brake,
        )
        self.cmd = result
        return result

    def reset(self) -> None:
        self.lateral.reset()
        self.speed.reset()
        self._steering_feedback = 0.0
        self._throttle_feedback = 0.0
        self._brake_feedback = 0.0
        self._gear_feedback = 1
        self.last_reference = None
        self.last_track_state = None
        self.last_safe_command = VehicleCommand()
        self.cmd = A2RLAckermannCommand()
