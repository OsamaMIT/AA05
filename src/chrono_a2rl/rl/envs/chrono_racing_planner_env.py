"""Gymnasium environment for an RL planner above the tracking controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chrono_a2rl.chrono_interface.direct_backend import ChronoDirectBackend
from chrono_a2rl.chrono_interface.reset_manager import initial_state_from_track
from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.common.math_utils import clamp
from chrono_a2rl.common.types import ControllerReference, VehicleCommand
from chrono_a2rl.control.mpc_lateral import LateralMPCController
from chrono_a2rl.control.reference_generator import make_reference
from chrono_a2rl.control.safety_supervisor import SafetySupervisor
from chrono_a2rl.control.speed_pid import SpeedPIDController
from chrono_a2rl.evaluation.metrics import MPS_TO_KMH
from chrono_a2rl.rl.corner_progress import CornerProgressTracker
from chrono_a2rl.rl.frontier import (
    EVALUATION_ROLE,
    FRONTIER_PRACTICE_ROLE,
    RANDOM_ROLE,
    START_LINE_ROLE,
)
from chrono_a2rl.rl.observations import (
    LONGITUDINAL_OBSERVATION_SIZE,
    make_longitudinal_observation,
)
from chrono_a2rl.rl.reset_sampling import sample_initial_state
from chrono_a2rl.rl.rewards import (
    compute_corner_braking_reward,
    compute_reward,
    compute_terminal_reward,
)
from chrono_a2rl.rl.trail_braking import (
    TrailBrakingReference,
    compute_trail_braking_reference,
)
from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_loader import load_track_from_config


class ChronoRacingPlannerEnv(gym.Env):
    """RL controls longitudinal pedals while MPC follows the fixed raceline."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path = "configs/experiments/rl_planner_yas_marina.yaml",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or load_experiment_config(config_path)
        self.track = load_track_from_config(self.config["track"])
        self.speed_profile = generate_speed_profile(self.track, self.config.get("speed_profile", {}))
        self.vehicle = self.config["vehicle"]
        self.simulation = self.config["simulation"]
        self.dt = float(self.simulation.get("control_dt", 0.02))
        self.max_episode_time = float(self.simulation.get("max_episode_time", 180.0))
        termination_cfg = self.config.get("termination", {})
        self.start_finish_s = self.track.wrap_s(
            float(termination_cfg.get("start_finish_s", 0.0))
        )
        self.minimum_lap_fraction = clamp(
            float(termination_cfg.get("minimum_lap_fraction", 0.95)),
            0.0,
            1.0,
        )
        rl_cfg = self.config.get("rl", {})
        reward_cfg = self.config.get("reward", {})
        self.training_role = str(rl_cfg.get("training_role", EVALUATION_ROLE))
        self.env_index = int(rl_cfg.get("env_index", -1))
        self.action_mode = str(rl_cfg.get("action_mode", "profile_pedal_residual")).lower()
        if self.action_mode != "profile_pedal_residual":
            raise ValueError(
                "ChronoRacingPlannerEnv requires action_mode=profile_pedal_residual. "
                "Start a fresh policy because the action contract has changed."
            )
        self.profile_tracking_enabled = bool(
            rl_cfg.get("profile_speed_tracking_enabled", True)
        )
        self.profile_residual_authority = clamp(
            float(rl_cfg.get("profile_speed_residual_authority", 0.08)),
            0.0,
            1.0,
        )
        self.profile_residual_error_guard = max(
            float(rl_cfg.get("profile_speed_residual_error_guard_mps", 5.0)),
            1.0e-6,
        )
        self.profile_residual_disable_during_coast = bool(
            rl_cfg.get("profile_residual_disable_during_coast", True)
        )
        self.longitudinal_action_deadband = clamp(
            float(rl_cfg.get("longitudinal_action_deadband", 0.05)),
            0.0,
            0.95,
        )
        self.longitudinal_action_rise_rate = max(
            float(rl_cfg.get("longitudinal_action_rise_rate", 6.0)),
            0.0,
        )
        self.longitudinal_action_fall_rate = max(
            float(rl_cfg.get("longitudinal_action_fall_rate", 10.0)),
            0.0,
        )
        self.lateral_reference = str(rl_cfg.get("lateral_offset_reference", "raceline")).lower()
        self.reference_lookahead_time = float(rl_cfg.get("reference_lookahead_time", 0.25))
        self.speed_floor_lateral_error_scale = float(
            rl_cfg.get("speed_floor_lateral_error_scale", 1.5)
        )
        self.speed_floor_heading_error_scale = float(
            rl_cfg.get("speed_floor_heading_error_scale", 0.25)
        )
        self.speed_floor_lookahead_distances = tuple(
            float(value)
            for value in rl_cfg.get("speed_floor_lookahead_distances", [0.0, 40.0, 80.0, 120.0])
        )
        self.speed_demand_curvature_source = str(
            rl_cfg.get("speed_demand_curvature_source", "raceline")
        ).lower()
        self.speed_demand_max_decel = float(
            rl_cfg.get(
                "speed_demand_max_decel",
                self.config.get("speed_profile", {}).get("max_decel", 16.0),
            )
        )
        self.speed_demand_brake_buffer = float(
            rl_cfg.get("speed_demand_brake_buffer", 10.0)
        )
        self.speed_demand_transition = float(
            rl_cfg.get("speed_demand_transition", 15.0)
        )
        self.speed_demand_full_reduction = float(
            rl_cfg.get("speed_demand_full_reduction", 25.0)
        )
        self.frontier_progress_m = float(rl_cfg.get("frontier_initial_progress_m", 350.0))
        self.frontier_clearance_m = float(rl_cfg.get("frontier_validation_clearance_m", 40.0))
        self.frontier_extension_cap_m = float(rl_cfg.get("frontier_max_advance_m", 150.0))
        self.frontier_target_distance = float("inf")
        self.frontier_advancement_m = 0.0
        self.frontier_cleared = False
        self.validated_progress_m = 0.0
        self.previous_validated_progress_m = 0.0
        self.episode_start_s = 0.0
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(LONGITUDINAL_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.backend = ChronoDirectBackend(self.vehicle, self.simulation)
        self.lateral = LateralMPCController(self.config["controller"].get("lateral", {}), self.vehicle)
        self.speed = SpeedPIDController(self.config["controller"].get("speed", {}))
        self.supervisor = SafetySupervisor(self.vehicle, self.simulation)
        self.corner_tracker = CornerProgressTracker(self.track, self.speed_profile, reward_cfg)
        self.state = self.backend.get_state()
        self.previous_s = 0.0
        self.progress_s = 0.0
        self.previous_pedal_action = 0.0
        self.effective_pedal_action = 0.0
        self.reference = ControllerReference()
        self.trail_braking_reference = TrailBrakingReference()
        self.step_count = 0
        self.curb_sample_count = 0
        self.curb_streak_steps = 0
        self.max_curb_streak_steps = 0
        self.last_corner_score = 0.0
        self.no_progress_steps = 0
        self.last_termination_reason = "running"

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        self.lateral.reset()
        self.speed.reset()
        self.progress_s = 0.0
        self.previous_pedal_action = 0.0
        self.effective_pedal_action = 0.0
        self.step_count = 0
        self.curb_sample_count = 0
        self.curb_streak_steps = 0
        self.max_curb_streak_steps = 0
        self.frontier_cleared = False
        self.frontier_advancement_m = 0.0
        self.validated_progress_m = 0.0
        self.previous_validated_progress_m = 0.0
        self.last_corner_score = 0.0
        self.no_progress_steps = 0
        self.last_termination_reason = "running"
        initial_state = self._sample_initial_state()
        self.state = self.backend.reset(initial_state)
        track_state = self.track.track_state_at_pose(self.state.x, self.state.y, self.state.yaw)
        self.previous_s = track_state.s
        self.episode_start_s = track_state.s
        self.frontier_target_distance = self._episode_frontier_target(track_state.s)
        base_lateral_offset = self._base_lateral_offset(track_state.s)
        lookahead_s = self._reference_s(track_state.s, 1.0)
        self.reference = make_reference(
            self.track,
            self.speed_profile,
            track_state,
            lateral_offset=base_lateral_offset,
            lookahead_lateral_offset=self._base_lateral_offset(lookahead_s),
            path_source=self.lateral_reference,
            horizon_steps=self.lateral.horizon_steps,
            control_dt=self.dt,
        )
        self.corner_tracker.reset(self.state, track_state)
        self.trail_braking_reference = self._trail_braking_reference(track_state.s)
        return self._observation(track_state), {}

    def step(self, action):
        action_array = np.asarray(action, dtype=float).reshape(-1)
        if action_array.size != 1:
            raise ValueError(
                "Profile residual policy expects one signed pedal action in [-1, 1]."
            )
        pedal_action = clamp(float(action_array[0]), -1.0, 1.0)
        previous_pedal_action = self.previous_pedal_action
        effective_pedal_action = self._rate_limited_pedal_action(pedal_action)
        policy_throttle, policy_brake = self._pedal_targets(effective_pedal_action)
        trail_braking_reference = self.trail_braking_reference
        previous_speed = self.state.speed
        track_state = self.track.track_state_at_pose(self.state.x, self.state.y, self.state.yaw)
        speed_alignment_strength = self._speed_alignment_strength(track_state)
        reference_s = self._reference_s(track_state.s, 1.0)
        base_lateral_offset = self._base_lateral_offset(track_state.s)
        lookahead_lateral_offset = self._base_lateral_offset(reference_s)
        self.reference = make_reference(
            self.track,
            self.speed_profile,
            track_state,
            lookahead_time=self.reference_lookahead_time,
            speed_scale=1.0,
            lateral_offset=base_lateral_offset,
            lookahead_lateral_offset=lookahead_lateral_offset,
            path_source=self.lateral_reference,
            horizon_steps=self.lateral.horizon_steps,
            control_dt=self.dt,
        )
        steer_cmd = self.lateral.compute_command(self.state, track_state, self.reference, self.dt)
        profile_speed_cmd = self.speed.compute_command(
            self.state,
            track_state,
            self.reference,
            self.dt,
        )
        (
            requested_throttle,
            requested_brake,
            effective_residual_pedal,
            residual_guard,
        ) = self._profile_tracking_targets(
            profile_speed_cmd,
            policy_throttle,
            policy_brake,
        )
        command = VehicleCommand(
            steering_target=steer_cmd.steering_target,
            throttle_target=requested_throttle,
            brake_target=requested_brake,
            gear_request=max(1, self.state.gear),
            command_timestamp=self.state.sim_time,
            command_valid_until=self.state.sim_time + self.dt,
        )
        safe_command = self.supervisor.supervise(command, self.state, track_state, self.dt)
        self.state = self.backend.step(safe_command, self.dt)
        new_track_state = self.track.track_state_at_pose(self.state.x, self.state.y, self.state.yaw)
        previous_track_s = self.previous_s
        progress_delta = self._progress_delta(new_track_state.s)
        self.progress_s += progress_delta
        self.previous_s = new_track_state.s
        self.previous_pedal_action = pedal_action
        self.effective_pedal_action = effective_pedal_action
        reward_cfg = self.config.get("reward", {})
        stall_progress_threshold = max(
            float(reward_cfg.get("stall_progress_threshold", 0.01)),
            0.0,
        )
        if progress_delta < stall_progress_threshold:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0
        no_progress_time = self.no_progress_steps * self.dt
        stall_timeout = max(float(reward_cfg.get("stall_timeout", 4.0)), self.dt)
        curb_usage_fraction, curb_streak_time = self._update_curb_stats(new_track_state.on_curb)
        racing_line_offset = self._base_lateral_offset(new_track_state.s)
        reference_racing_line_offset = self._base_lateral_offset(self.reference.target_s)
        lateral_offset_delta = 0.0
        apex_strength = self._corner_range_strength(new_track_state.s)
        (
            new_speed_corner_strength,
            future_curve_speed_cap,
            speed_demand_distance,
            required_braking_distance,
        ) = self._speed_braking_demand(
            new_track_state.s,
        )
        offtrack = not new_track_state.on_track
        completed = self._lap_completed(
            previous_track_s,
            new_track_state.s,
            on_track=not offtrack,
        )
        crossed_start_finish = self.track.crossed_line_forward(
            previous_track_s,
            new_track_state.s,
            line_s=self.start_finish_s,
        )
        stalled = no_progress_time >= stall_timeout and not offtrack and not completed
        if not offtrack:
            self.validated_progress_m = (
                self.track.length
                if completed
                else max(0.0, self.progress_s - self.frontier_clearance_m)
            )
        reward = compute_reward(
            progress_delta=progress_delta,
            state=self.state,
            track_state=new_track_state,
            command=safe_command,
            target_speed=self.reference.target_speed,
            racing_line_offset=racing_line_offset,
            reference_racing_line_offset=racing_line_offset,
            target_lateral_offset=self.reference.target_lateral_offset,
            lateral_offset_delta=lateral_offset_delta,
            apex_strength=apex_strength,
            speed_pressure_strength=1.0 - new_speed_corner_strength,
            curb_usage_fraction=curb_usage_fraction,
            curb_streak_time=curb_streak_time,
            config=self.config.get("reward", {}),
        )
        actual_deceleration = max(
            0.0,
            (previous_speed - self.state.speed) / max(self.dt, 1.0e-6),
        )
        corner_braking = compute_corner_braking_reward(
            speed=self.state.speed,
            future_speed_cap=future_curve_speed_cap,
            braking_demand=new_speed_corner_strength,
            actual_deceleration=actual_deceleration,
            brake_command=self.state.brake,
            trail_brake_target=trail_braking_reference.target_brake,
            line_safety=speed_alignment_strength,
            config=reward_cfg,
        )
        corner_braking_reward_scale = float(
            reward_cfg.get(
                "corner_braking_reward_scale",
                0.0 if self.profile_tracking_enabled else 1.0,
            )
        )
        applied_corner_braking_reward = (
            corner_braking_reward_scale * corner_braking.total
        )
        reward += applied_corner_braking_reward
        pedal_change = abs(pedal_action - previous_pedal_action)
        pedal_change_penalty = float(
            self.config.get("reward", {}).get(
                "longitudinal_action_change_penalty",
                0.0,
            )
        ) * pedal_change
        reward -= pedal_change_penalty
        corner_update = self.corner_tracker.update(
            self.state,
            new_track_state,
            progress_delta,
        )
        self.trail_braking_reference = self._trail_braking_reference(new_track_state.s)
        self.last_corner_score = corner_update.score
        reward += corner_update.reward
        frontier_reward, frontier_shortfall_penalty = self._update_frontier_reward(
            terminal=completed or offtrack or stalled,
        )
        reward += frontier_reward
        terminated = completed or offtrack or stalled
        truncated = self.state.sim_time >= self.max_episode_time and not terminated
        if truncated and not completed:
            _, frontier_shortfall_penalty = self._update_frontier_reward(terminal=True)
            reward -= frontier_shortfall_penalty
        elif terminated:
            reward -= frontier_shortfall_penalty
        speed_fraction = self.state.speed / max(
            float(reward_cfg.get("max_reward_speed", 83.3333333333)),
            1.0,
        )
        kinetic_crash_penalty = (
            float(reward_cfg.get("offtrack_kinetic_penalty", 0.0))
            * clamp(speed_fraction, 0.0, 1.5) ** 2
            if offtrack
            else 0.0
        )
        corner_failure = offtrack and corner_update.in_corner_or_clearance
        terminal_reward = 0.0
        if terminated or truncated:
            terminal_reward = compute_terminal_reward(
                completed=completed,
                offtrack=offtrack,
                timeout=truncated and not completed,
                stalled=stalled,
                progress_fraction=self.progress_s / max(self.track.length, 1.0),
                speed_fraction=speed_fraction,
                corner_failure=corner_failure,
                curb_usage_fraction=curb_usage_fraction,
                max_curb_streak_time=self.max_curb_streak_steps * self.dt,
                config=reward_cfg,
            )
            reward += terminal_reward
        if completed:
            self.last_termination_reason = "lap_completed"
        elif offtrack:
            self.last_termination_reason = "off_track"
        elif stalled:
            self.last_termination_reason = "stalled"
        elif truncated:
            self.last_termination_reason = "timeout"
        else:
            self.last_termination_reason = "running"
        info = {
            "progress_s": self.progress_s,
            "progress_fraction": self.progress_s / max(self.track.length, 1.0),
            "validated_progress_m": self.validated_progress_m,
            "frontier_progress_m": self.frontier_progress_m,
            "frontier_advancement_m": self.frontier_advancement_m,
            "frontier_target_distance_m": self.frontier_target_distance,
            "frontier_cleared": self.frontier_cleared,
            "frontier_reward": frontier_reward,
            "frontier_shortfall_penalty": frontier_shortfall_penalty,
            "training_role": self.training_role,
            "env_index": self.env_index,
            "track_s": new_track_state.s,
            "start_finish_s": self.start_finish_s,
            "crossed_start_finish": crossed_start_finish,
            "on_track": new_track_state.on_track,
            "on_curb": new_track_state.on_curb,
            "curb_side": new_track_state.curb_side,
            "curb_penalty_weight": new_track_state.curb_penalty_weight,
            "curb_usage_fraction": curb_usage_fraction,
            "curb_streak_time": curb_streak_time,
            "target_speed_kmh": self.reference.target_speed * MPS_TO_KMH,
            "target_lateral_offset": self.reference.target_lateral_offset,
            "racing_line_offset": racing_line_offset,
            "reference_s": self.reference.target_s,
            "reference_racing_line_offset": reference_racing_line_offset,
            "controller_racing_line_offset": self.reference.target_lateral_offset,
            "strategy_lateral_offset": 0.0,
            "strategy_lateral_offset_delta": lateral_offset_delta,
            "apex_strength": apex_strength,
            "corner_id": corner_update.corner_id,
            "corner_phase": corner_update.phase,
            "corner_completed": corner_update.completed,
            "corner_failed": corner_update.failed,
            "corner_completion_count": self.corner_tracker.completion_count,
            "corner_completion_reward": corner_update.reward,
            "corner_score": corner_update.score,
            "corner_progress": corner_update.distance,
            "corner_distance_completion": corner_update.distance_completion,
            "corner_heading_completion": corner_update.heading_completion,
            "distance_to_apex": self.corner_tracker.distance_to_apex(new_track_state.s),
            "apex_passed": corner_update.apex_passed,
            "apex_quality": corner_update.apex_quality,
            "apex_speed_kmh": corner_update.apex_speed * MPS_TO_KMH,
            "exit_speed_kmh": corner_update.exit_speed * MPS_TO_KMH,
            "corner_speed_quality": corner_update.speed_quality,
            "corner_failure_active": corner_failure,
            "speed_corner_strength": new_speed_corner_strength,
            "future_raceline_speed_cap_kmh": future_curve_speed_cap * MPS_TO_KMH,
            "speed_demand_distance_m": speed_demand_distance,
            "required_braking_distance_m": required_braking_distance,
            "actual_deceleration": actual_deceleration,
            "corner_overspeed_fraction": corner_braking.overspeed_fraction,
            "corner_controlled_braking_reward": (
                corner_braking.controlled_braking_reward
            ),
            "corner_overspeed_penalty": corner_braking.overspeed_penalty,
            "corner_excessive_braking_penalty": (
                corner_braking.excessive_braking_penalty
            ),
            "corner_underspeed_penalty": corner_braking.underspeed_penalty,
            "corner_braking_reward_applied": applied_corner_braking_reward,
            "trail_braking_active": trail_braking_reference.active,
            "trail_braking_phase": trail_braking_reference.phase,
            "trail_braking_corner_id": trail_braking_reference.corner_id,
            "trail_brake_target": trail_braking_reference.target_brake,
            "trail_brake_command": trail_braking_reference.target_brake,
            "trail_brake_applied": self.state.brake,
            "trail_brake_alignment_error": corner_braking.alignment_error,
            "trail_brake_alignment_reward": corner_braking.alignment_reward,
            "trail_brake_missing_penalty": corner_braking.missing_brake_penalty,
            "trail_brake_excess_reference_penalty": (
                corner_braking.excess_reference_penalty
            ),
            "trail_brake_release_quality": corner_braking.release_quality,
            "trail_brake_required_deceleration": (
                trail_braking_reference.required_deceleration
            ),
            "trail_brake_coast_deceleration": (
                trail_braking_reference.coast_deceleration
            ),
            "speed_alignment_strength": speed_alignment_strength,
            "floor_speed_scale": 1.0,
            "desired_speed_scale": 1.0,
            "speed_action": pedal_action,
            "requested_speed_scale": 1.0,
            "raw_speed_scale": 1.0,
            "effective_speed_scale": 1.0,
            "speed_scale": 1.0,
            "lateral_offset_fraction": 0.0,
            "action_mode": self.action_mode,
            "longitudinal_action": pedal_action,
            "effective_longitudinal_action": effective_pedal_action,
            "longitudinal_action_change": pedal_change,
            "longitudinal_action_change_penalty": pedal_change_penalty,
            "no_progress_time": no_progress_time,
            "stalled": stalled,
            "termination_reason": self.last_termination_reason,
            "requested_throttle": requested_throttle,
            "requested_brake": requested_brake,
            "policy_requested_throttle": policy_throttle,
            "policy_requested_brake": policy_brake,
            "profile_tracking_enabled": self.profile_tracking_enabled,
            "profile_speed_error_kmh": (
                self.state.speed - self.reference.target_speed
            )
            * MPS_TO_KMH,
            "profile_pid_throttle": profile_speed_cmd.throttle_target,
            "profile_pid_brake": profile_speed_cmd.brake_target,
            "profile_pid_mode": self.speed.last_mode,
            "profile_residual_pedal": effective_residual_pedal,
            "profile_residual_guard": residual_guard,
            "profile_residual_authority": self.profile_residual_authority,
            "commanded_throttle": safe_command.throttle_target,
            "commanded_brake": safe_command.brake_target,
            "applied_throttle": self.state.throttle,
            "applied_brake": self.state.brake,
            "steering_target": steer_cmd.steering_target,
            "lateral_controller_mode": self.lateral.mode,
            "mpc_solver_status": self.lateral.last_solver_status,
            "mpc_nominal_steering": self.lateral.last_nominal_steering,
            "mpc_ancillary_correction": self.lateral.last_ancillary_correction,
            "mpc_tube_lateral_bound": float(self.lateral.last_tube_state_bound[0]),
            "mpc_tube_heading_bound": float(self.lateral.last_tube_state_bound[1]),
            "mpc_tube_input_bound": self.lateral.last_tube_input_bound,
            "kinetic_crash_penalty": kinetic_crash_penalty,
            "terminal_reward": terminal_reward,
            "supervisor_reason": self.supervisor.last_reason,
        }
        self.frontier_advancement_m = 0.0
        return self._observation(new_track_state), float(reward), bool(terminated), bool(truncated), info

    def _progress_delta(self, new_s: float) -> float:
        ds = new_s - self.previous_s
        if ds < -0.5 * self.track.length:
            ds += self.track.length
        elif ds > 0.5 * self.track.length:
            ds -= self.track.length
        return max(0.0, ds)

    def _lap_completed(
        self,
        previous_s: float,
        current_s: float,
        *,
        on_track: bool,
    ) -> bool:
        """Require a real forward start/finish crossing for lap completion."""

        return bool(
            on_track
            and self.progress_s >= self.minimum_lap_fraction * self.track.length
            and self.track.crossed_line_forward(
                previous_s,
                current_s,
                line_s=self.start_finish_s,
            )
        )

    def _update_curb_stats(self, on_curb: bool) -> tuple[float, float]:
        self.step_count += 1
        if on_curb:
            self.curb_sample_count += 1
            self.curb_streak_steps += 1
            self.max_curb_streak_steps = max(self.max_curb_streak_steps, self.curb_streak_steps)
        else:
            self.curb_streak_steps = 0
        return (
            self.curb_sample_count / max(1, self.step_count),
            self.curb_streak_steps * self.dt,
        )

    def set_progress_frontier(self, progress_m: float) -> None:
        """Receive a synchronized curriculum frontier from the training callback."""

        next_frontier = clamp(float(progress_m), 0.0, self.track.length)
        self.frontier_advancement_m += max(0.0, next_frontier - self.frontier_progress_m)
        self.frontier_progress_m = next_frontier
        self.frontier_target_distance = self._episode_frontier_target(self.episode_start_s)

    def get_progress_frontier(self) -> float:
        """Return the local copy of the synchronized curriculum frontier."""

        return self.frontier_progress_m

    def get_track_length(self) -> float:
        """Return track length without exposing the full geometry cross-process."""

        return self.track.length

    def _sample_initial_state(self):
        rl_cfg = self.config.get("rl", {})
        if (
            self.training_role == EVALUATION_ROLE
            and not bool(rl_cfg.get("randomize_resets", False))
            and bool(rl_cfg.get("eval_start_at_profile_speed", True))
        ):
            return initial_state_from_track(
                self.track,
                s=0.0,
                speed=self.speed_profile.speed_at(0.0),
                lateral_offset=self._base_lateral_offset(0.0),
                heading_source=self.lateral_reference,
            )

        if self.training_role == START_LINE_ROLE:
            return initial_state_from_track(
                self.track,
                s=0.0,
                speed=float(self.simulation.get("initial_speed", 12.0)),
                lateral_offset=self._base_lateral_offset(0.0),
                heading_source=self.lateral_reference,
            )

        if self.training_role == FRONTIER_PRACTICE_ROLE:
            lookback_min = float(rl_cfg.get("frontier_practice_lookback_min_m", 150.0))
            lookback_max = float(rl_cfg.get("frontier_practice_lookback_max_m", 250.0))
            lookback = float(self.np_random.uniform(lookback_min, max(lookback_min, lookback_max)))
            start_s = self.track.wrap_s(self.frontier_progress_m - lookback)
            scale_min = float(rl_cfg.get("reset_speed_scale_min", 0.60))
            scale_max = float(rl_cfg.get("reset_speed_scale_max", 1.10))
            speed_scale = float(self.np_random.uniform(scale_min, max(scale_min, scale_max)))
            speed = min(
                self.speed_profile.speed_at(start_s) * speed_scale,
                float(self.vehicle.get("max_speed", self.speed_profile.max_speed)),
            )
            heading_limit = float(rl_cfg.get("reset_heading_error_max", 0.08))
            heading_error = float(self.np_random.uniform(-heading_limit, heading_limit))
            return initial_state_from_track(
                self.track,
                s=start_s,
                speed=speed,
                lateral_offset=self._base_lateral_offset(start_s),
                heading_error=heading_error,
                heading_source=self.lateral_reference,
            )

        sample_cfg = dict(rl_cfg)
        if self.training_role == RANDOM_ROLE:
            sample_cfg["randomize_resets"] = True
            sample_cfg["reset_s_mode"] = "random"
        return sample_initial_state(
            track=self.track,
            simulation_config=self.simulation,
            rl_config=sample_cfg,
            rng=self.np_random,
            speed_profile=self.speed_profile,
            vehicle_config=self.vehicle,
        )

    def _episode_frontier_target(self, start_s: float) -> float:
        frontier_s = self.track.wrap_s(self.frontier_progress_m)
        if self.training_role == START_LINE_ROLE:
            return min(self.frontier_progress_m, self.track.length)
        if self.training_role == FRONTIER_PRACTICE_ROLE:
            return float((frontier_s - start_s) % self.track.length)
        return float("inf")

    def _update_frontier_reward(self, *, terminal: bool) -> tuple[float, float]:
        if not np.isfinite(self.frontier_target_distance):
            self.previous_validated_progress_m = self.validated_progress_m
            return 0.0, 0.0

        cfg = self.config.get("reward", {})
        target = self.frontier_target_distance
        previous_extension = clamp(
            self.previous_validated_progress_m - target,
            0.0,
            self.frontier_extension_cap_m,
        )
        current_extension = clamp(
            self.validated_progress_m - target,
            0.0,
            self.frontier_extension_cap_m,
        )
        reward = float(cfg.get("frontier_progress_weight", 2.0)) * max(
            0.0,
            current_extension - previous_extension,
        )
        if not self.frontier_cleared and self.validated_progress_m >= target:
            self.frontier_cleared = True
            reward += float(cfg.get("frontier_clear_bonus", 300.0))

        shortfall_penalty = 0.0
        if terminal and not self.frontier_cleared:
            shortfall_scale = max(float(cfg.get("frontier_shortfall_scale_m", 150.0)), 1.0)
            shortfall_penalty = float(cfg.get("frontier_shortfall_penalty", 500.0)) * clamp(
                (target - self.validated_progress_m) / shortfall_scale,
                0.0,
                1.0,
            )
        self.previous_validated_progress_m = self.validated_progress_m
        return reward, shortfall_penalty

    def _pedal_targets(self, action: float) -> tuple[float, float]:
        """Map one signed policy action to mutually exclusive pedals."""

        pedal = clamp(action, -1.0, 1.0)
        magnitude = abs(pedal)
        if magnitude <= self.longitudinal_action_deadband:
            return 0.0, 0.0
        normalized = (magnitude - self.longitudinal_action_deadband) / (
            1.0 - self.longitudinal_action_deadband
        )
        if pedal > 0.0:
            return normalized, 0.0
        return 0.0, normalized

    def _rate_limited_pedal_action(self, requested_action: float) -> float:
        """Filter abrupt brake/throttle reversals before DBW actuation."""

        requested = clamp(requested_action, -1.0, 1.0)
        delta = requested - self.effective_pedal_action
        rate = (
            self.longitudinal_action_rise_rate
            if delta >= 0.0
            else self.longitudinal_action_fall_rate
        )
        if rate <= 0.0:
            return requested
        max_delta = rate * self.dt
        return self.effective_pedal_action + clamp(delta, -max_delta, max_delta)

    def _profile_tracking_targets(
        self,
        profile_command: VehicleCommand,
        policy_throttle: float,
        policy_brake: float,
    ) -> tuple[float, float, float, float]:
        """Blend a bounded PPO pedal residual into authoritative profile PID."""

        if not self.profile_tracking_enabled:
            return policy_throttle, policy_brake, policy_throttle - policy_brake, 1.0

        speed_error = abs(self.reference.target_speed - self.state.speed)
        residual_guard = clamp(
            1.0 - speed_error / self.profile_residual_error_guard,
            0.0,
            1.0,
        )
        policy_pedal = policy_throttle - policy_brake
        effective_residual = (
            policy_pedal
            * self.profile_residual_authority
            * residual_guard
        )
        if (
            self.profile_residual_disable_during_coast
            and self.speed.last_mode == "coast"
        ):
            effective_residual = 0.0
        profile_pedal = (
            profile_command.throttle_target - profile_command.brake_target
        )
        combined_pedal = clamp(profile_pedal + effective_residual, -1.0, 1.0)
        return (
            max(0.0, combined_pedal),
            max(0.0, -combined_pedal),
            effective_residual,
            residual_guard,
        )

    def _base_lateral_offset(self, s: float) -> float:
        if self.lateral_reference == "centerline":
            return 0.0
        return self.track.raceline_lateral_offset_at(s)

    def _reference_s(self, s: float, speed_scale: float) -> float:
        target_speed = self.speed_profile.target_speed_at(s, speed_scale)
        return self.track.wrap_s(s + max(target_speed, 0.0) * self.reference_lookahead_time)

    def _speed_alignment_strength(self, track_state: TrackState) -> float:
        line_error_scale = max(self.speed_floor_lateral_error_scale, 1.0e-6)
        heading_error_scale = max(self.speed_floor_heading_error_scale, 1.0e-6)
        line_error = abs(track_state.n - self._base_lateral_offset(track_state.s))
        line_strength = clamp(1.0 - line_error / line_error_scale, 0.0, 1.0)
        heading_strength = clamp(1.0 - abs(track_state.heading_error) / heading_error_scale, 0.0, 1.0)
        return min(line_strength, heading_strength)

    def _speed_braking_demand(self, s: float) -> tuple[float, float, float, float]:
        """Return braking demand from speed-feasible future raceline bends.

        The straight speed floor is released only when a sampled future bend
        requires less than maximum speed and lies inside its estimated braking
        window. This keeps shallow, speed-feasible bends flat-out and makes the
        trigger distance proportional to the actual raceline speed reduction.
        """

        profile_cfg = self.config.get("speed_profile", {})
        max_speed = self.speed_profile.max_speed
        min_speed = self.speed_profile.min_speed
        max_lateral_accel = max(float(profile_cfg.get("max_lateral_accel", 18.0)), 1.0e-6)
        epsilon = max(float(profile_cfg.get("curvature_epsilon", 1.0e-4)), 1.0e-9)
        max_decel = max(self.speed_demand_max_decel, 1.0e-6)
        transition = max(self.speed_demand_transition, 1.0e-6)
        full_reduction = max(self.speed_demand_full_reduction, 1.0e-6)

        strongest_demand = 0.0
        critical_cap = max_speed
        critical_distance = 0.0
        critical_braking_distance = 0.0
        for distance in self.speed_floor_lookahead_distances:
            lookahead = max(0.0, float(distance))
            curvature = abs(
                self.track.curvature_at(
                    s + lookahead,
                    source=self.speed_demand_curvature_source,
                )
            )
            curve_cap = clamp(
                float(np.sqrt(max_lateral_accel / max(curvature, epsilon))),
                min_speed,
                max_speed,
            )
            speed_reduction = max(0.0, max_speed - curve_cap)
            if speed_reduction <= 0.0:
                continue
            braking_distance = max(
                0.0,
                (max_speed**2 - curve_cap**2) / (2.0 * max_decel),
            )
            release_start = (
                braking_distance
                + max(0.0, self.speed_demand_brake_buffer)
                + transition
            )
            proximity = clamp((release_start - lookahead) / transition, 0.0, 1.0)
            severity = clamp(speed_reduction / full_reduction, 0.0, 1.0)
            demand = proximity * severity
            if demand > strongest_demand:
                strongest_demand = demand
                critical_cap = curve_cap
                critical_distance = lookahead
                critical_braking_distance = braking_distance
        return (
            strongest_demand,
            critical_cap,
            critical_distance,
            critical_braking_distance,
        )

    def _corner_range_strength(
        self,
        s: float,
        *,
        distances: tuple[float, ...] | list[float] | None = None,
        threshold: float | None = None,
        full: float | None = None,
    ) -> float:
        reward_cfg = self.config.get("reward", {})
        threshold_value = (
            float(reward_cfg.get("apex_curvature_threshold", 0.015))
            if threshold is None
            else float(threshold)
        )
        full_value = (
            float(reward_cfg.get("apex_curvature_full", 0.06))
            if full is None
            else float(full)
        )
        full_value = max(full_value, threshold_value + 1.0e-6)
        distances = distances or reward_cfg.get("apex_lookahead_distances", [0.0, 20.0, 45.0, 70.0])
        max_strength = 0.0
        for distance in distances:
            curvature = self.track.interpolate(s + float(distance)).curvature
            strength = clamp(
                (abs(curvature) - threshold_value) / (full_value - threshold_value),
                0.0,
                1.0,
            )
            max_strength = max(max_strength, strength)
        return max_strength

    def _trail_braking_reference(self, s: float) -> TrailBrakingReference:
        return compute_trail_braking_reference(
            tracker=self.corner_tracker,
            speed_profile=self.speed_profile,
            s=s,
            speed=self.state.speed,
            vehicle_config=self.vehicle,
            config=self.config.get("reward", {}),
        )

    def _observation(self, track_state):
        frontier_distance = (
            self.frontier_target_distance - self.validated_progress_m
            if np.isfinite(self.frontier_target_distance)
            else 0.0
        )
        return make_longitudinal_observation(
            self.state,
            track_state,
            self.reference,
            self.track,
            self.speed_profile,
            previous_pedal_action=self.effective_pedal_action,
            progress_s=self.progress_s,
            frontier_distance=frontier_distance,
            corner_phase=self.corner_tracker.phase_value,
            distance_to_apex=self.corner_tracker.distance_to_apex(track_state.s),
            corner_heading_completion=self.corner_tracker.heading_completion,
            trail_brake_target=self.trail_braking_reference.target_brake,
            curvature_source=self.speed_demand_curvature_source,
        )

    def render(self):
        return None

    def close(self) -> None:
        self.backend.close()
