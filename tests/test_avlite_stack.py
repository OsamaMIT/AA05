"""Contract tests for the AVLite migration boundary."""

import pytest

from avlite.c10_perception.c11_perception_model import Map, RaceMap
from avlite.c20_planning.c21_planning_model import GlobalPlan

from chrono_a2rl.avlite_stack.backend_adapter import AVLiteBackendAdapter
from chrono_a2rl.avlite_stack.planning import A2RLGlobalRacePlanner
from chrono_a2rl.avlite_stack.runtime import (
    DEFAULT_PROFILE,
    _bootstrap_avlite_race_plot,
    load_profile,
    validate_stack,
)
from chrono_a2rl.common.config import REPO_ROOT, load_experiment_config
from chrono_a2rl.common.types import VehicleCommand
from chrono_a2rl.rl.envs.chrono_racing_planner_env import ChronoRacingPlannerEnv


def _mock_config() -> dict:
    config = load_experiment_config(
        "configs/experiments/rl_planner_yas_marina.yaml"
    )
    config["simulation"]["backend"] = "mock"
    return config


def test_exported_yas_marina_is_avlite_native() -> None:
    plan_path = REPO_ROOT / "avlite_data/yas_marina_a2rl_global_plan.json"
    map_path = REPO_ROOT / "avlite_data/yas_marina_a2rl.map.json"

    assert GlobalPlan.is_loadable(plan_path)
    plan = GlobalPlan.from_file(str(plan_path))
    race_map = Map.open(map_path)

    assert len(plan.path) == 1110
    assert len(plan.velocity) == len(plan.path)
    assert max(plan.velocity) * 3.6 == pytest.approx(300.0)
    assert isinstance(race_map, RaceMap)
    assert len(race_map.left_bound) == len(plan.path)
    assert len(race_map.right_bound) == len(plan.path)


def test_global_planner_preserves_raceline_limits_and_speed() -> None:
    planner = A2RLGlobalRacePlanner()
    plan = planner.plan()

    assert len(plan.path) == len(planner.track.s_nodes)
    assert len(plan.left_boundary_d) == len(plan.path)
    assert len(plan.right_boundary_d) == len(plan.path)
    assert min(plan.left_boundary_d) > 0.0
    assert max(plan.right_boundary_d) < 0.0
    assert max(plan.velocity) * 3.6 == pytest.approx(300.0)


def test_backend_adapter_updates_avlite_ego_in_place() -> None:
    backend = AVLiteBackendAdapter(_mock_config())
    try:
        ego = backend.world.ego_state
        state = backend.reset()
        next_state = backend.step(
            VehicleCommand(
                throttle_target=0.5,
                gear_request=1,
                command_timestamp=state.sim_time,
                command_valid_until=state.sim_time + 0.02,
            ),
            0.02,
        )

        assert backend.world.ego_state is ego
        assert ego.x == pytest.approx(next_state.x)
        assert ego.y == pytest.approx(next_state.y)
        assert ego.velocity == pytest.approx(next_state.speed)
    finally:
        backend.close()


def test_gym_environment_uses_avlite_world_bridge() -> None:
    env = ChronoRacingPlannerEnv(config=_mock_config())
    try:
        observation, _ = env.reset(seed=4)
        _, _, terminated, truncated, _ = env.step([0.0])

        assert isinstance(env.backend, AVLiteBackendAdapter)
        assert observation.shape == env.observation_space.shape
        assert not (terminated and truncated)
    finally:
        env.close()


def test_upstream_avlite_factory_builds_complete_a2rl_stack() -> None:
    result = validate_stack(backend="mock")

    assert result["executor"] == "SyncExecuter"
    assert result["world"] == "A2RLChronoWorldBridge"
    assert result["global_planner"] == "A2RLGlobalRacePlanner"
    assert result["local_planner"] == "A2RLReferenceLocalPlanner"
    assert result["controller"] == "A2RLMPCControlStrategy"
    assert result["sim_time_s"] == pytest.approx(0.02)


def test_visualizer_bootstrap_uses_race_canvas_then_restores_profile() -> None:
    from avlite.c40_execution.c49_settings import ExecutionSettings
    from avlite.c60_apps.c62_factory import load_stack_settings

    load_profile(DEFAULT_PROFILE, backend="mock")
    assert ExecutionSettings.c40_global_planner == "A2RLGlobalRacePlanner"

    configured = _bootstrap_avlite_race_plot()
    assert configured == "A2RLGlobalRacePlanner"
    assert ExecutionSettings.c40_global_planner == "GlobalCenterlineRacePlanner"

    # This mirrors VisualizerApp.load_settings(), which runs before executor
    # construction and restores the actual custom planner from the profile.
    load_stack_settings(profile=DEFAULT_PROFILE, load_plugins=True)
    assert ExecutionSettings.c40_global_planner == "A2RLGlobalRacePlanner"
