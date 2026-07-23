"""Reward functions for the first high-level speed policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chrono_a2rl.common.math_utils import clamp
from chrono_a2rl.common.types import TrackState, VehicleCommand, VehicleState


@dataclass(slots=True)
class CornerBrakingReward:
    """Reward breakdown for speed-controlled corner-entry braking."""

    total: float = 0.0
    overspeed_fraction: float = 0.0
    controlled_braking_reward: float = 0.0
    overspeed_penalty: float = 0.0
    excessive_braking_penalty: float = 0.0
    underspeed_penalty: float = 0.0
    alignment_reward: float = 0.0
    alignment_error: float = 0.0
    missing_brake_penalty: float = 0.0
    excess_reference_penalty: float = 0.0
    release_quality: float = 1.0


def compute_corner_braking_reward(
    *,
    speed: float,
    future_speed_cap: float,
    braking_demand: float,
    actual_deceleration: float,
    brake_command: float,
    trail_brake_target: float = 0.0,
    line_safety: float,
    config: dict[str, Any] | None = None,
) -> CornerBrakingReward:
    """Reward controlled deceleration when a future bend requires it.

    The term acts only inside the curvature-derived braking window. It rewards
    achieved moderate brake-assisted deceleration while penalizing both
    excessive corner-entry speed and unnecessarily hard stops. Coasting does
    not earn controlled-braking credit.
    """

    cfg = config or {}
    demand = clamp(braking_demand, 0.0, 1.0)
    target_brake = clamp(trail_brake_target, 0.0, 1.0)
    applied_brake = clamp(brake_command, 0.0, 1.0)
    max_reference = max(
        float(cfg.get("trail_braking_max_reference", 0.70)),
        1.0e-6,
    )
    shaping_strength = max(
        demand,
        clamp(target_brake / max_reference, 0.0, 1.0),
    )
    maximum_speed = max(float(cfg.get("max_reward_speed", 83.3333333333)), 1.0)
    speed_cap = clamp(float(future_speed_cap), 0.0, maximum_speed)
    if demand <= 0.0 and target_brake <= 0.0:
        return CornerBrakingReward(
            alignment_error=applied_brake,
            release_quality=1.0 - applied_brake,
        )

    cap_margin = max(float(cfg.get("corner_speed_cap_margin_fraction", 0.08)), 0.0)
    allowed_speed = min(maximum_speed, speed_cap * (1.0 + cap_margin))
    overspeed_scale = max(maximum_speed - allowed_speed, maximum_speed * 0.10)
    overspeed = clamp((speed - allowed_speed) / overspeed_scale, 0.0, 1.5)

    target_deceleration = max(
        float(cfg.get("corner_braking_target_decel", 6.0)),
        1.0e-6,
    )
    maximum_deceleration = max(
        float(cfg.get("corner_braking_max_decel", 10.0)),
        target_deceleration,
    )
    achieved_deceleration = max(0.0, float(actual_deceleration))
    controlled_fraction = clamp(
        achieved_deceleration / target_deceleration,
        0.0,
        1.0,
    )
    brake_assistance = (
        clamp(applied_brake / max(target_brake, 0.05), 0.0, 1.0)
        if target_brake > 0.0
        else 0.0
    )
    controlled_reward = (
        float(cfg.get("corner_controlled_braking_reward_weight", 0.0))
        * shaping_strength
        * overspeed
        * controlled_fraction
        * brake_assistance
        * clamp(line_safety, 0.0, 1.0)
    )
    overspeed_penalty = (
        float(cfg.get("corner_overspeed_penalty_weight", 0.0))
        * demand
        * overspeed**2
    )

    excessive_deceleration = clamp(
        (achieved_deceleration - maximum_deceleration) / maximum_deceleration,
        0.0,
        1.0,
    )
    maximum_brake = clamp(
        float(cfg.get("corner_braking_max_command", 0.75)),
        0.0,
        1.0,
    )
    excessive_brake = clamp(
        (max(0.0, brake_command) - maximum_brake)
        / max(1.0 - maximum_brake, 1.0e-6),
        0.0,
        1.0,
    )
    excessive_penalty = (
        float(cfg.get("corner_excessive_braking_penalty_weight", 0.0))
        * shaping_strength
        * max(excessive_deceleration, excessive_brake) ** 2
    )

    minimum_speed_fraction = clamp(
        float(cfg.get("corner_min_speed_cap_fraction", 0.72)),
        0.0,
        1.0,
    )
    minimum_speed = speed_cap * minimum_speed_fraction
    underspeed_fraction = clamp(
        (minimum_speed - speed) / max(minimum_speed, 1.0),
        0.0,
        1.0,
    )
    underspeed_penalty = (
        float(cfg.get("corner_underspeed_penalty_weight", 0.0))
        * demand
        * underspeed_fraction**2
    )
    alignment_error = abs(applied_brake - target_brake)
    reference_active = target_brake > 1.0e-6
    alignment_quality = (
        clamp(
            1.0 - alignment_error / max(target_brake, 0.05),
            0.0,
            1.0,
        )
        if reference_active
        else 0.0
    )
    alignment_reward = (
        float(cfg.get("trail_braking_alignment_reward_weight", 0.35))
        * (target_brake / max_reference) ** 0.5
        * alignment_quality
        * clamp(line_safety, 0.0, 1.0)
    )
    missing_fraction = (
        clamp(
            (target_brake - applied_brake) / max(target_brake, 1.0e-6),
            0.0,
            1.0,
        )
        if reference_active
        else 0.0
    )
    missing_brake_penalty = (
        float(cfg.get("trail_braking_missing_penalty_weight", 1.25))
        * missing_fraction
    )
    excess_fraction = (
        clamp(
            (applied_brake - target_brake) / max(1.0 - target_brake, 1.0e-6),
            0.0,
            1.0,
        )
        if reference_active
        else 0.0
    )
    excess_reference_penalty = (
        float(cfg.get("trail_braking_excess_reference_penalty_weight", 0.75))
        * excess_fraction**2
    )
    release_quality = alignment_quality if reference_active else 1.0 - applied_brake
    return CornerBrakingReward(
        total=(
            controlled_reward
            + alignment_reward
            - overspeed_penalty
            - excessive_penalty
            - underspeed_penalty
            - missing_brake_penalty
            - excess_reference_penalty
        ),
        overspeed_fraction=overspeed,
        controlled_braking_reward=controlled_reward,
        overspeed_penalty=overspeed_penalty,
        excessive_braking_penalty=excessive_penalty,
        underspeed_penalty=underspeed_penalty,
        alignment_reward=alignment_reward,
        alignment_error=alignment_error,
        missing_brake_penalty=missing_brake_penalty,
        excess_reference_penalty=excess_reference_penalty,
        release_quality=release_quality,
    )


def compute_reward(
    *,
    progress_delta: float,
    state: VehicleState,
    track_state: TrackState,
    command: VehicleCommand,
    target_speed: float | None = None,
    racing_line_offset: float = 0.0,
    reference_racing_line_offset: float | None = None,
    target_lateral_offset: float = 0.0,
    lateral_offset_delta: float = 0.0,
    apex_strength: float = 0.0,
    speed_pressure_strength: float | None = None,
    curb_usage_fraction: float = 0.0,
    curb_streak_time: float = 0.0,
    config: dict[str, Any] | None = None,
) -> float:
    """Compute dense racing reward.

    Progress remains the main signal. Additional terms reward achieved speed
    and clean line tracking while penalizing time, curb overuse, instability,
    and excessive actuator effort.
    """

    cfg = config or {}
    on_track_gate = 1.0 if track_state.on_track else 0.0
    progress_gate_distance = max(
        float(cfg.get("positive_reward_progress_distance", 0.20)),
        1.0e-6,
    )
    motion_gate = clamp(progress_delta / progress_gate_distance, 0.0, 1.0)
    reward = float(cfg.get("progress_weight", 1.0)) * progress_delta * on_track_gate
    max_reward_speed = max(
        float(cfg.get("max_reward_speed", target_speed or 83.3333333333)),
        1.0,
    )

    speed_fraction = clamp(state.speed / max_reward_speed, 0.0, 1.5)
    reward += (
        float(cfg.get("speed_reward_weight", 0.0))
        * speed_fraction
        * motion_gate
        * on_track_gate
    )

    target_fraction = 0.0
    if target_speed is not None:
        target_fraction = clamp(target_speed / max_reward_speed, 0.0, 1.5)
        tracking_fraction = clamp(state.speed / max(target_speed, 1.0), 0.0, 1.2)
        achieved_target_fraction = min(target_fraction, speed_fraction)
        reward += (
            float(cfg.get("target_speed_reward_weight", 0.0))
            * achieved_target_fraction
            * motion_gate
            * on_track_gate
        )
        profile_error_scale = max(
            float(cfg.get("profile_speed_error_scale_kmh", 15.0)) / 3.6,
            1.0e-6,
        )
        profile_error_fraction = clamp(
            abs(state.speed - target_speed) / profile_error_scale,
            0.0,
            2.0,
        )
        reward += (
            float(cfg.get("profile_speed_tracking_reward_weight", 0.0))
            * clamp(1.0 - profile_error_fraction, 0.0, 1.0)
            * motion_gate
            * on_track_gate
        )
        reward -= (
            float(cfg.get("profile_speed_error_penalty_weight", 0.0))
            * profile_error_fraction**2
            * motion_gate
        )
        reward += (
            float(cfg.get("target_speed_tracking_weight", 0.0))
            * tracking_fraction
            * motion_gate
            * on_track_gate
        )

    min_reward_speed = float(cfg.get("min_reward_speed", 0.0))
    if min_reward_speed > 0.0 and state.speed < min_reward_speed:
        reward -= float(cfg.get("low_speed_penalty", 0.0)) * (
            (min_reward_speed - state.speed) / min_reward_speed
        )

    if float(cfg.get("lateral_exploration_weight", 0.0)) > 0.0:
        usable_width = max(
            track_state.distance_left_boundary + track_state.distance_right_boundary,
            1.0,
        )
        lateral_fraction = clamp(abs(track_state.n) / usable_width, 0.0, 1.0)
        reward += (
            float(cfg.get("lateral_exploration_weight", 0.0))
            * lateral_fraction
            * motion_gate
        )

    line_error_scale = max(float(cfg.get("racing_line_error_scale", 1.5)), 1.0e-6)
    line_error = track_state.n - racing_line_offset
    line_error_fraction = clamp(abs(line_error) / line_error_scale, 0.0, 2.0)
    line_accuracy = clamp(1.0 - line_error_fraction, 0.0, 1.0)
    reward -= float(cfg.get("racing_line_error_penalty", 0.0)) * line_error_fraction**2

    target_offset_scale = max(float(cfg.get("target_offset_error_scale", 1.0)), 1.0e-6)
    reference_line = racing_line_offset if reference_racing_line_offset is None else reference_racing_line_offset
    target_offset_error = target_lateral_offset - reference_line
    target_offset_fraction = clamp(abs(target_offset_error) / target_offset_scale, 0.0, 2.0)
    reward -= float(cfg.get("target_offset_penalty", 0.0)) * target_offset_fraction**2
    reward -= float(cfg.get("target_offset_change_penalty", 0.0)) * (
        abs(lateral_offset_delta) / target_offset_scale
    )

    safe_speed_margin = max(float(cfg.get("safe_speed_boundary_margin", 1.5)), 1.0e-6)
    safe_speed_boundary = clamp(
        min(
            track_state.distance_left_boundary,
            track_state.distance_right_boundary,
        )
        / safe_speed_margin,
        0.0,
        1.0,
    )
    reward += (
        float(cfg.get("safe_speed_progress_weight", 0.0))
        * progress_delta
        * speed_fraction
        * line_accuracy
        * safe_speed_boundary
        * on_track_gate
    )

    apex = clamp(apex_strength, 0.0, 1.0)
    straight = (
        clamp(speed_pressure_strength, 0.0, 1.0)
        if speed_pressure_strength is not None
        else 1.0 - apex
    )
    if straight > 0.0:
        reward += (
            float(cfg.get("straight_speed_reward_weight", 0.0))
            * straight
            * speed_fraction
            * motion_gate
            * on_track_gate
        )
        if target_speed is not None:
            reward += (
                float(cfg.get("straight_target_speed_reward_weight", 0.0))
                * straight
                * min(target_fraction, speed_fraction)
                * motion_gate
                * on_track_gate
            )
            min_target_fraction = float(cfg.get("straight_min_target_speed_fraction", 0.0))
            if min_target_fraction > 0.0 and target_fraction < min_target_fraction:
                reward -= (
                    float(cfg.get("straight_low_target_speed_penalty", 0.0))
                    * straight
                    * ((min_target_fraction - target_fraction) / min_target_fraction)
                )
        min_actual_fraction = float(cfg.get("straight_min_actual_speed_fraction", 0.0))
        if min_actual_fraction > 0.0 and speed_fraction < min_actual_fraction:
            reward -= (
                float(cfg.get("straight_low_actual_speed_penalty", 0.0))
                * straight
                * ((min_actual_fraction - speed_fraction) / min_actual_fraction)
            )

    if apex > 0.0:
        reward -= float(cfg.get("corner_line_error_penalty", 0.0)) * apex * line_error_fraction**2
        reward += (
            float(cfg.get("apex_line_reward_weight", 0.0))
            * apex
            * line_accuracy
            * motion_gate
        )
        nearest_boundary = min(
            track_state.distance_left_boundary,
            track_state.distance_right_boundary,
        )
        safe_margin = max(float(cfg.get("apex_boundary_margin", 1.0)), 1.0e-6)
        boundary_safety = clamp(nearest_boundary / safe_margin, 0.0, 1.0)
        if boundary_safety < 1.0:
            reward -= float(cfg.get("apex_boundary_penalty", 0.0)) * apex * (1.0 - boundary_safety)
        reward += (
            float(cfg.get("apex_speed_reward_weight", 0.0))
            * apex
            * speed_fraction
            * line_accuracy
            * boundary_safety
            * motion_gate
        )

    reward -= float(cfg.get("time_penalty_per_step", 0.0))
    reward -= float(cfg.get("stationary_penalty_per_step", 0.0)) * (1.0 - motion_gate)

    if not track_state.on_track:
        reward -= float(cfg.get("offtrack_penalty", 25.0))
    if track_state.on_curb:
        curb_weight = max(0.0, track_state.curb_penalty_weight)
        reward -= float(cfg.get("curb_penalty_scale", 1.0)) * curb_weight
        reward -= (
            float(cfg.get("curb_high_speed_penalty_scale", 0.0))
            * curb_weight
            * speed_fraction
        )
        boundary_margin = float(cfg.get("curb_near_boundary_margin", 1.0))
        if boundary_margin > 0.0:
            nearest_boundary = min(
                track_state.distance_left_boundary,
                track_state.distance_right_boundary,
            )
            boundary_fraction = clamp((boundary_margin - nearest_boundary) / boundary_margin, 0.0, 1.0)
            reward -= float(cfg.get("curb_near_boundary_penalty_scale", 0.0)) * curb_weight * boundary_fraction

    overuse_limit = float(cfg.get("curb_overuse_fraction_limit", 1.0))
    if track_state.on_curb and curb_usage_fraction > overuse_limit:
        overuse_penalty = float(cfg.get("curb_overuse_penalty_scale", 0.0)) * (
            curb_usage_fraction - overuse_limit
        )
        reward -= min(
            overuse_penalty,
            float(cfg.get("curb_overuse_penalty_cap", overuse_penalty)),
        )

    streak_limit = float(cfg.get("curb_streak_time_limit", 1.0e9))
    if track_state.on_curb and curb_streak_time > streak_limit:
        streak_penalty = float(cfg.get("curb_streak_penalty_scale", 0.0)) * (
            curb_streak_time - streak_limit
        )
        reward -= min(
            streak_penalty,
            float(cfg.get("curb_streak_penalty_cap", streak_penalty)),
        )

    if abs(state.yaw_rate) > float(cfg.get("max_abs_yaw_rate", 2.5)):
        reward -= float(cfg.get("instability_penalty", 10.0))
    control_effort = abs(command.steering_target) + command.throttle_target + command.brake_target
    reward -= float(cfg.get("excessive_control_penalty", 0.05)) * control_effort
    return reward


def compute_terminal_reward(
    *,
    completed: bool,
    offtrack: bool,
    timeout: bool,
    stalled: bool = False,
    progress_fraction: float,
    speed_fraction: float = 0.0,
    corner_failure: bool = False,
    curb_usage_fraction: float = 0.0,
    max_curb_streak_time: float = 0.0,
    config: dict[str, Any] | None = None,
) -> float:
    """Sparse terminal shaping for completion, crashes, stalls, and timeout."""

    cfg = config or {}
    reward = 0.0
    if completed:
        reward += float(cfg.get("completion_bonus", 0.0))
    if offtrack:
        reward -= float(cfg.get("offtrack_terminal_penalty", 0.0))
        remaining_fraction = clamp(1.0 - progress_fraction, 0.0, 1.0)
        reward -= float(cfg.get("offtrack_remaining_lap_penalty", 0.0)) * remaining_fraction
        reward -= float(cfg.get("offtrack_speed_penalty", 0.0)) * clamp(
            speed_fraction,
            0.0,
            1.5,
        )
        reward -= float(cfg.get("offtrack_kinetic_penalty", 0.0)) * clamp(
            speed_fraction,
            0.0,
            1.5,
        ) ** 2
        if corner_failure:
            reward -= float(cfg.get("offtrack_corner_penalty", 0.0))
    if timeout:
        remaining_fraction = clamp(1.0 - progress_fraction, 0.0, 1.0)
        reward -= float(cfg.get("timeout_penalty", 0.0)) * remaining_fraction
    if stalled:
        reward -= float(cfg.get("stall_terminal_penalty", 0.0))
    curb_limit = float(cfg.get("curb_overuse_fraction_limit", 1.0))
    if curb_usage_fraction > curb_limit:
        normalized_overuse = (curb_usage_fraction - curb_limit) / max(1.0 - curb_limit, 1.0e-6)
        reward -= float(cfg.get("curb_overuse_terminal_penalty", 0.0)) * clamp(
            normalized_overuse,
            0.0,
            1.0,
        )
    streak_limit = float(cfg.get("curb_streak_time_limit", 1.0e9))
    if max_curb_streak_time > streak_limit:
        reward -= float(cfg.get("curb_streak_terminal_penalty", 0.0)) * clamp(
            (max_curb_streak_time - streak_limit) / max(streak_limit, 1.0),
            0.0,
            1.0,
        )
    return reward
