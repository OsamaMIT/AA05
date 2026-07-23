from __future__ import annotations

from pathlib import Path

from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.rl.envs.chrono_racing_planner_env import ChronoRacingPlannerEnv
from chrono_a2rl.rl.frontier import (
    FRONTIER_PRACTICE_ROLE,
    RANDOM_ROLE,
    START_LINE_ROLE,
    ProgressFrontierState,
    assign_training_role,
    load_frontier_state,
    save_frontier_state,
)
from chrono_a2rl.rl.run_manager import (
    create_run_directory,
    resolve_latest_model,
    resolve_latest_run,
)


def test_default_parallel_role_assignment() -> None:
    roles = [assign_training_role(index, 8) for index in range(8)]

    assert roles == [
        START_LINE_ROLE,
        START_LINE_ROLE,
        FRONTIER_PRACTICE_ROLE,
        FRONTIER_PRACTICE_ROLE,
        FRONTIER_PRACTICE_ROLE,
        FRONTIER_PRACTICE_ROLE,
        RANDOM_ROLE,
        RANDOM_ROLE,
    ]


def test_frontier_advancement_is_monotonic_and_capped() -> None:
    state = ProgressFrontierState(frontier_progress_m=350.0)

    assert state.advance(700.0, maximum_advance_m=150.0, track_length=1000.0)
    assert state.frontier_progress_m == 500.0
    assert state.best_validated_progress_m == 700.0
    assert not state.advance(450.0, maximum_advance_m=150.0, track_length=1000.0)
    assert state.frontier_progress_m == 500.0


def test_frontier_state_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint_frontier.yaml"
    expected = ProgressFrontierState(
        run_id="test_run",
        frontier_progress_m=725.0,
        best_validated_progress_m=800.0,
        update_count=3,
        total_timesteps=12000,
    )

    save_frontier_state(path, expected)
    actual = load_frontier_state(path)

    assert actual == expected


def test_latest_run_and_model_resolution(tmp_path: Path) -> None:
    first_id, first = create_run_directory(tmp_path, seed=1)
    assert first_id
    (first / "final_model.zip").write_text("first")
    second_id, second = create_run_directory(tmp_path, seed=2)
    assert second_id != first_id
    checkpoint = second / "ppo_planner_frontier_100_steps.zip"
    checkpoint.write_text("second")

    assert resolve_latest_run(tmp_path) == second
    assert resolve_latest_model(tmp_path) == checkpoint


def test_frontier_practice_reset_starts_before_target() -> None:
    config = load_experiment_config("configs/experiments/rl_planner_yas_marina.yaml")
    config["simulation"]["backend"] = "mock"
    config["rl"]["training_role"] = FRONTIER_PRACTICE_ROLE
    config["rl"]["frontier_initial_progress_m"] = 350.0
    env = ChronoRacingPlannerEnv(config=config)

    env.reset(seed=4)

    assert 150.0 <= env.frontier_target_distance <= 250.0
    assert 100.0 <= env.episode_start_s <= 200.0
    env.close()


def test_frontier_reward_requires_validated_clearance() -> None:
    config = load_experiment_config("configs/experiments/rl_planner_yas_marina.yaml")
    config["simulation"]["backend"] = "mock"
    config["rl"]["training_role"] = START_LINE_ROLE
    env = ChronoRacingPlannerEnv(config=config)
    env.reset(seed=2)

    env.validated_progress_m = 349.0
    before, _ = env._update_frontier_reward(terminal=False)
    env.validated_progress_m = 350.0
    cleared, _ = env._update_frontier_reward(terminal=False)

    assert before == 0.0
    assert cleared == config["reward"]["frontier_clear_bonus"]
    assert env.frontier_cleared
    env.close()


def test_environment_reports_synchronized_frontier_advancement_once() -> None:
    config = load_experiment_config("configs/experiments/rl_planner_yas_marina.yaml")
    config["simulation"]["backend"] = "mock"
    config["rl"]["training_role"] = START_LINE_ROLE
    env = ChronoRacingPlannerEnv(config=config)
    env.reset(seed=3)

    env.set_progress_frontier(425.0)
    _, _, terminated, truncated, info = env.step([0.0])

    assert not terminated
    assert not truncated
    assert env.frontier_target_distance == 425.0
    assert info["frontier_advancement_m"] == 75.0
    assert env.frontier_advancement_m == 0.0
    env.close()
