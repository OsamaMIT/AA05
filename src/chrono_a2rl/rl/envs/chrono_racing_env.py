"""Gymnasium environment wrapping the MPC baseline and direct backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chrono_a2rl.chrono_interface.direct_backend import ChronoDirectBackend
from chrono_a2rl.chrono_interface.reset_manager import initial_state_from_track
from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.common.types import ControllerReference, VehicleCommand
from chrono_a2rl.control.mpc_lateral import LateralMPCController
from chrono_a2rl.control.reference_generator import make_reference
from chrono_a2rl.control.safety_supervisor import SafetySupervisor
from chrono_a2rl.control.speed_pid import SpeedPIDController
from chrono_a2rl.rl.observations import make_observation
from chrono_a2rl.rl.rewards import compute_reward
from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_loader import load_track_from_config


class ChronoRacingEnv(gym.Env):
    """High-level racing environment where RL scales target speed."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_path: str | Path = "configs/experiments/rl_speed_policy_yas_marina.yaml",
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
        rl_cfg = self.config.get("rl", {})
        self.action_space = spaces.Box(
            low=np.array([float(rl_cfg.get("target_speed_scale_min", 0.65))], dtype=np.float32),
            high=np.array([float(rl_cfg.get("target_speed_scale_max", 1.15))], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32,
        )
        self.backend = ChronoDirectBackend(self.vehicle, self.simulation)
        self.lateral = LateralMPCController(self.config["controller"].get("lateral", {}), self.vehicle)
        self.speed = SpeedPIDController(self.config["controller"].get("speed", {}))
        self.supervisor = SafetySupervisor(self.vehicle, self.simulation)
        self.state = self.backend.get_state()
        self.previous_s = 0.0
        self.progress_s = 0.0
        self.previous_action = 1.0
        self.reference = ControllerReference()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options
        self.lateral.reset()
        self.speed.reset()
        self.progress_s = 0.0
        self.previous_action = 1.0
        initial_speed = float(self.simulation.get("initial_speed", 3.0))
        initial_state = initial_state_from_track(self.track, s=0.0, speed=initial_speed)
        self.state = self.backend.reset(initial_state)
        track_state = self.track.track_state_at_pose(self.state.x, self.state.y, self.state.yaw)
        self.previous_s = track_state.s
        self.reference = make_reference(self.track, self.speed_profile, track_state)
        obs = make_observation(
            self.state,
            track_state,
            self.reference,
            previous_action=self.previous_action,
            progress_s=self.progress_s,
        )
        return obs, {}

    def step(self, action):
        scale = float(np.asarray(action, dtype=float).reshape(-1)[0])
        track_state = self.track.track_state_at_pose(self.state.x, self.state.y, self.state.yaw)
        self.reference = make_reference(
            self.track,
            self.speed_profile,
            track_state,
            speed_scale=scale,
        )
        steer_cmd = self.lateral.compute_command(self.state, track_state, self.reference, self.dt)
        speed_cmd = self.speed.compute_command(self.state, track_state, self.reference, self.dt)
        command = VehicleCommand(
            steering_target=steer_cmd.steering_target,
            throttle_target=speed_cmd.throttle_target,
            brake_target=speed_cmd.brake_target,
            gear_request=speed_cmd.gear_request,
            command_timestamp=self.state.sim_time,
            command_valid_until=self.state.sim_time + self.dt,
        )
        safe_command = self.supervisor.supervise(command, self.state, track_state, self.dt)
        self.state = self.backend.step(safe_command, self.dt)
        new_track_state = self.track.track_state_at_pose(self.state.x, self.state.y, self.state.yaw)
        ds = new_track_state.s - self.previous_s
        if ds < -0.5 * self.track.length:
            ds += self.track.length
        elif ds > 0.5 * self.track.length:
            ds -= self.track.length
        progress_delta = max(0.0, ds)
        self.progress_s += progress_delta
        self.previous_s = new_track_state.s
        self.previous_action = scale
        obs = make_observation(
            self.state,
            new_track_state,
            self.reference,
            previous_action=self.previous_action,
            progress_s=self.progress_s,
        )
        reward = compute_reward(
            progress_delta=progress_delta,
            state=self.state,
            track_state=new_track_state,
            command=safe_command,
            config=self.config.get("reward", {}),
        )
        terminated = self.progress_s >= 0.98 * self.track.length or not new_track_state.on_track
        truncated = self.state.sim_time >= self.max_episode_time
        info = {
            "progress_s": self.progress_s,
            "track_s": new_track_state.s,
            "on_track": new_track_state.on_track,
            "on_curb": new_track_state.on_curb,
            "curb_side": new_track_state.curb_side,
            "curb_penalty_weight": new_track_state.curb_penalty_weight,
            "target_speed": self.reference.target_speed,
            "supervisor_reason": self.supervisor.last_reason,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return None

    def close(self) -> None:
        self.backend.close()
