"""Reward functions for the first high-level speed policy."""

from __future__ import annotations

from typing import Any

from chrono_a2rl.common.types import TrackState, VehicleCommand, VehicleState


def compute_reward(
    *,
    progress_delta: float,
    state: VehicleState,
    track_state: TrackState,
    command: VehicleCommand,
    config: dict[str, Any] | None = None,
) -> float:
    """Progress reward minus off-track, instability, and control penalties."""

    cfg = config or {}
    reward = float(cfg.get("progress_weight", 1.0)) * progress_delta
    if not track_state.on_track:
        reward -= float(cfg.get("offtrack_penalty", 25.0))
    if track_state.on_curb:
        reward -= float(cfg.get("curb_penalty_scale", 1.0)) * track_state.curb_penalty_weight
    if abs(state.yaw_rate) > float(cfg.get("max_abs_yaw_rate", 2.5)):
        reward -= float(cfg.get("instability_penalty", 10.0))
    control_effort = abs(command.steering_target) + command.throttle_target + command.brake_target
    reward -= float(cfg.get("excessive_control_penalty", 0.05)) * control_effort
    return reward
