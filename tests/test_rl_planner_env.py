from __future__ import annotations

import numpy as np
import pytest

from chrono_a2rl.chrono_interface.reset_manager import initial_state_from_track
from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.common.types import VehicleCommand
from chrono_a2rl.rl.envs.chrono_racing_planner_env import ChronoRacingPlannerEnv
from chrono_a2rl.rl.observations import LONGITUDINAL_OBSERVATION_SIZE


def _make_env() -> ChronoRacingPlannerEnv:
    config = load_experiment_config("configs/experiments/rl_planner_yas_marina.yaml")
    config["simulation"]["backend"] = "mock"
    config["rl"]["randomize_resets"] = False
    return ChronoRacingPlannerEnv(config=config)


def test_longitudinal_env_reset_and_step_mock_backend() -> None:
    env = _make_env()
    assert np.allclose(env.action_space.low, [-1.0])
    assert np.allclose(env.action_space.high, [1.0])
    obs, info = env.reset(seed=3)
    assert obs.shape == (LONGITUDINAL_OBSERVATION_SIZE,)
    assert info == {}

    next_obs, reward, terminated, truncated, info = env.step([0.0])

    assert next_obs.shape == (LONGITUDINAL_OBSERVATION_SIZE,)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert info["action_mode"] == "profile_pedal_residual"
    assert info["policy_requested_throttle"] == 0.0
    assert info["policy_requested_brake"] == 0.0
    assert info["profile_tracking_enabled"] is True
    assert "trail_brake_target" in info
    assert "trail_brake_alignment_error" in info
    assert np.isclose(next_obs[-1], env.trail_braking_reference.target_brake)
    env.close()


def test_lap_requires_start_finish_crossing_not_just_98_percent_progress() -> None:
    env = _make_env()
    env.progress_s = 0.98 * env.track.length

    assert not env._lap_completed(
        0.97 * env.track.length,
        0.98 * env.track.length,
        on_track=True,
    )

    env.progress_s = env.track.length
    assert env._lap_completed(
        env.track.length - 1.0,
        1.0,
        on_track=True,
    )
    assert not env._lap_completed(
        env.track.length - 1.0,
        1.0,
        on_track=False,
    )
    env.close()


@pytest.mark.parametrize(
    ("action", "throttle", "brake"),
    [
        (1.0, 1.0, 0.0),
        (-1.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.04, 0.0, 0.0),
    ],
)
def test_signed_pedal_mapping_is_mutually_exclusive(
    action: float,
    throttle: float,
    brake: float,
) -> None:
    env = _make_env()

    actual_throttle, actual_brake = env._pedal_targets(action)

    assert np.isclose(actual_throttle, throttle)
    assert np.isclose(actual_brake, brake)
    assert actual_throttle == 0.0 or actual_brake == 0.0
    env.close()


def test_policy_action_cannot_change_mpc_steering_reference() -> None:
    throttle_env = _make_env()
    brake_env = _make_env()
    throttle_env.reset(seed=8)
    brake_env.reset(seed=8)

    _, _, _, _, throttle_info = throttle_env.step([1.0])
    _, _, _, _, brake_info = brake_env.step([-1.0])

    assert np.isclose(
        throttle_info["steering_target"],
        brake_info["steering_target"],
    )
    assert np.isclose(
        throttle_info["target_lateral_offset"],
        brake_info["target_lateral_offset"],
    )
    assert throttle_info["strategy_lateral_offset"] == 0.0
    assert brake_info["strategy_lateral_offset"] == 0.0
    throttle_env.close()
    brake_env.close()


def test_longitudinal_observation_contains_actuator_feedback_and_previous_action() -> None:
    env = _make_env()
    env.reset(seed=1)

    obs, _, _, _, info = env.step([1.0])

    assert np.isclose(obs[11], info["applied_throttle"])
    assert np.isclose(obs[12], info["applied_brake"])
    assert np.isclose(obs[13], info["effective_longitudinal_action"])
    assert np.isclose(info["effective_longitudinal_action"], 0.12)
    env.close()


def test_brake_action_slows_more_than_throttle_action() -> None:
    throttle_env = _make_env()
    brake_env = _make_env()
    throttle_env.reset(seed=2)
    brake_env.reset(seed=2)
    profile_speed = throttle_env.speed_profile.speed_at(175.0)
    initial = initial_state_from_track(
        throttle_env.track,
        s=175.0,
        speed=profile_speed,
        lateral_offset=throttle_env._base_lateral_offset(175.0),
    )
    throttle_env.state = throttle_env.backend.reset(initial)
    brake_env.state = brake_env.backend.reset(initial)
    throttle_env.previous_s = 175.0
    brake_env.previous_s = 175.0

    for _ in range(10):
        throttle_env.step([1.0])
        brake_env.step([-1.0])

    assert throttle_env.state.speed > brake_env.state.speed
    assert throttle_env.state.throttle > 0.0
    assert brake_env.state.throttle < throttle_env.state.throttle
    throttle_env.close()
    brake_env.close()


def test_controller_tracks_current_raceline_with_separate_lookahead_preview() -> None:
    env = _make_env()
    env.reset(seed=4)

    _, _, _, _, info = env.step([0.7])

    assert np.isclose(
        info["controller_racing_line_offset"],
        info["target_lateral_offset"],
    )
    assert abs(info["target_lateral_offset"] - info["racing_line_offset"]) < 0.05
    assert info["lateral_offset_fraction"] == 0.0
    env.close()


def test_pedal_action_reversals_are_rate_limited() -> None:
    env = _make_env()
    env.reset(seed=7)

    _, _, _, _, throttle_info = env.step([1.0])
    _, _, _, _, brake_info = env.step([-1.0])

    assert np.isclose(throttle_info["effective_longitudinal_action"], 0.12)
    assert np.isclose(brake_info["effective_longitudinal_action"], -0.08)
    assert brake_info["policy_requested_throttle"] == 0.0
    assert brake_info["policy_requested_brake"] > 0.0
    env.close()


def test_old_two_action_checkpoint_contract_is_rejected() -> None:
    env = _make_env()
    env.reset()

    with pytest.raises(ValueError, match="expects one signed pedal action"):
        env.step([0.0, 0.0])

    env.close()


def test_stationary_policy_is_terminated_as_stalled() -> None:
    config = load_experiment_config("configs/experiments/rl_planner_yas_marina.yaml")
    config["simulation"]["backend"] = "mock"
    config["rl"]["randomize_resets"] = False
    config["reward"]["stall_timeout"] = 0.04
    env = ChronoRacingPlannerEnv(config=config)
    env.reset(seed=5)
    initial = initial_state_from_track(
        env.track,
        s=175.0,
        speed=0.0,
        lateral_offset=env._base_lateral_offset(175.0),
    )
    env.state = env.backend.reset(initial)
    env.previous_s = 175.0

    _, _, first_terminated, _, first_info = env.step([0.0])
    _, second_reward, second_terminated, _, second_info = env.step([0.0])

    assert not first_terminated
    assert first_info["no_progress_time"] == env.dt
    assert second_terminated
    assert second_info["stalled"] is True
    assert second_info["termination_reason"] == "stalled"
    assert second_info["terminal_reward"] == -600.0
    assert second_reward < -600.0
    env.close()


def test_raceline_braking_demand_remains_available_as_observation_diagnostic() -> None:
    env = _make_env()
    env.track.curvature_at = lambda _s, source="centerline": 0.001
    shallow_demand = env._speed_braking_demand(0.0)
    env.track.curvature_at = lambda _s, source="centerline": 0.02
    tight_demand = env._speed_braking_demand(0.0)

    assert shallow_demand[0] == 0.0
    assert np.isclose(shallow_demand[1], env.speed_profile.max_speed)
    assert tight_demand[0] > 0.9
    assert tight_demand[1] < env.speed_profile.max_speed
    env.close()


def test_trail_braking_target_is_observed_and_reported_before_low_speed_corner() -> None:
    env = _make_env()
    env.reset(seed=9)
    point = env.track.interpolate(1230.0)
    initial = initial_state_from_track(
        env.track,
        s=1230.0,
        speed=81.0,
        lateral_offset=env._base_lateral_offset(1230.0),
        heading_source=env.lateral_reference,
    )
    env.state = env.backend.reset(initial)
    track_state = env.track.track_state_at_pose(point.x, point.y, point.heading)
    env.corner_tracker.reset(env.state, track_state)
    env.previous_s = track_state.s
    env.trail_braking_reference = env._trail_braking_reference(track_state.s)

    observation = env._observation(track_state)
    _, _, _, _, info = env.step([0.0])

    assert observation[-1] > 0.0
    assert info["trail_braking_active"] is True
    assert info["trail_braking_corner_id"] == 2
    assert info["trail_brake_target"] > 0.0
    assert info["trail_brake_missing_penalty"] > 0.0
    env.close()


def test_profile_pid_removes_residual_authority_for_large_speed_error() -> None:
    env = _make_env()
    env.reset(seed=10)
    track_state = env.track.track_state_at_pose(
        env.state.x,
        env.state.y,
        env.state.yaw,
    )
    env.reference = env.reference.__class__(target_speed=env.state.speed + 20.0)
    profile_command = env.speed.compute_command(
        env.state,
        track_state,
        env.reference,
        env.dt,
    )

    throttle, brake, residual, guard = env._profile_tracking_targets(
        profile_command,
        policy_throttle=0.0,
        policy_brake=1.0,
    )

    assert guard == 0.0
    assert residual == 0.0
    assert np.isclose(throttle, profile_command.throttle_target)
    assert np.isclose(brake, profile_command.brake_target)
    env.close()


def test_profile_residual_is_bounded_near_target_speed() -> None:
    env = _make_env()
    env.reset(seed=11)
    env.reference = env.reference.__class__(target_speed=env.state.speed)

    throttle, brake, residual, guard = env._profile_tracking_targets(
        VehicleCommand(),
        policy_throttle=1.0,
        policy_brake=0.0,
    )

    assert guard == 1.0
    assert np.isclose(residual, 0.08)
    assert np.isclose(throttle, 0.08)
    assert brake == 0.0
    env.close()


def test_deterministic_evaluation_starts_at_profile_speed() -> None:
    env = _make_env()

    env.reset(seed=12)

    assert np.isclose(env.state.speed, env.speed_profile.speed_at(0.0))
    env.close()
