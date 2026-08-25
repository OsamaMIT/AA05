"""Runtime assembly for the repository-owned AVLite profile and plugin."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sysconfig
from typing import Any

from chrono_a2rl.avlite_stack.settings import A2RLStackSettings
from chrono_a2rl.common.config import REPO_ROOT, resolve_path


DEFAULT_PROFILE = "a2rl_chrono_yas_marina"


def configure_avlite_paths() -> None:
    """Point AVLite at the repository's portable profile, data, and plugin roots."""

    profile_dir = REPO_ROOT / "avlite_profiles"
    data_dir = REPO_ROOT / "avlite_data"
    if not profile_dir.is_dir() or not data_dir.is_dir():
        installed = (
            Path(sysconfig.get_path("data")) / "share" / "chrono_a2rl"
        )
        profile_dir = installed / "avlite_profiles"
        data_dir = installed / "avlite_data"

    plugin_spec = importlib.util.find_spec("avlite_a2rl")
    if plugin_spec is None or not plugin_spec.submodule_search_locations:
        raise RuntimeError("The packaged avlite_a2rl plugin cannot be located.")
    plugin_dir = Path(next(iter(plugin_spec.submodule_search_locations))).parent
    os.environ["AVLITE_CONFIG_DIR"] = str(profile_dir)
    os.environ["AVLITE_DATA_DIR"] = str(data_dir)
    os.environ["AVLITE_PLUGINS_DIR"] = str(plugin_dir)


def load_profile(
    profile: str = DEFAULT_PROFILE,
    *,
    experiment_config: str | None = None,
    backend: str | None = None,
) -> None:
    """Load AVLite layer settings plus this community plugin's settings."""

    configure_avlite_paths()
    from avlite.c60_apps.c62_factory import load_stack_settings
    from avlite.c60_apps.c63_plugins import (
        load_community_plugin_setting,
        sync_builtin_plugins,
    )
    from avlite.c60_apps.c69_settings import AppSettings

    # Establish the bundled plugin package path before community discovery adds
    # the repository plugin root to the same namespace.
    sync_builtin_plugins(
        ["p60_headless_mode", "p60_setting_cli", "p60_visualizer_tk"]
    )
    load_stack_settings(profile=profile, load_plugins=True)
    stored = AppSettings.c62_community_plugins.get("avlite_a2rl", "avlite_a2rl")
    load_community_plugin_setting("avlite_a2rl", stored, profile=profile)
    if experiment_config:
        A2RLStackSettings.experiment_config = str(resolve_path(experiment_config))
    if backend:
        A2RLStackSettings.backend = backend


def build_executor(
    profile: str = DEFAULT_PROFILE,
    *,
    experiment_config: str | None = None,
    backend: str | None = None,
):
    """Build the complete AVLite executor from the active repository profile."""

    load_profile(
        profile,
        experiment_config=experiment_config,
        backend=backend,
    )
    from avlite.c40_execution.c49_settings import ExecutionSettings
    from avlite.c60_apps.c62_factory import executor_factory

    return executor_factory(
        executer_type=ExecutionSettings.c40_executer_type,
        bridge=ExecutionSettings.c40_bridge,
        perception_strategy_name=ExecutionSettings.c40_perception,
        localization_strategy_name=ExecutionSettings.c40_localization,
        mapping_strategy_name=ExecutionSettings.c40_mapping,
        global_planner_strategy_name=ExecutionSettings.c40_global_planner,
        local_planner_strategy_name=ExecutionSettings.c40_local_planner,
        controller_strategy_name=ExecutionSettings.c40_controller,
        execution_task_names=ExecutionSettings.c40_execution_tasks,
        perception_dt=ExecutionSettings.c40_perception_dt,
        localization_dt=ExecutionSettings.c40_localization_dt,
        replan_dt=ExecutionSettings.c40_replan_dt,
        control_dt=ExecutionSettings.c40_control_dt,
        default_global_trajectory_file=ExecutionSettings.c40_global_trajectory,
        map_file=ExecutionSettings.c40_map,
        load_plugins=True,
        async_combined_perception_planning=False,
    )


def validate_stack(
    profile: str = DEFAULT_PROFILE,
    *,
    experiment_config: str | None = None,
    backend: str | None = "mock",
) -> dict[str, Any]:
    """Assemble, reset, and step the AVLite stack once."""

    executor = build_executor(
        profile,
        experiment_config=experiment_config,
        backend=backend,
    )
    try:
        executor.reset()
        _step_executor(executor)
        return {
            "profile": profile,
            "executor": type(executor).__name__,
            "world": type(executor.world).__name__,
            "global_planner": type(executor.global_planner).__name__,
            "local_planner": type(executor.local_planner).__name__,
            "controller": type(executor.controller).__name__,
            "sim_time_s": float(executor.elapsed_sim_time),
            "speed_kmh": float(executor.ego_state.velocity * 3.6),
        }
    finally:
        executor.world.close()


def run_stack(
    profile: str = DEFAULT_PROFILE,
    *,
    experiment_config: str | None = None,
    backend: str | None = None,
    max_sim_time: float | None = None,
) -> dict[str, Any]:
    """Run the synchronous AVLite stack to task completion."""

    from avlite.c40_execution.c43_task_strategy import StackEvent

    executor = build_executor(
        profile,
        experiment_config=experiment_config,
        backend=backend,
    )
    limit = float(
        max_sim_time
        if max_sim_time is not None
        else executor.world.simulation_config.get("max_episode_time", 300.0)
    )
    try:
        executor.reset()
        executor.task_runner.notify(StackEvent.EXECUTION_STARTED)
        next_report = 10.0
        while not executor.stopped and executor.elapsed_sim_time < limit:
            _step_executor(executor)
            if executor.elapsed_sim_time >= next_report:
                world = executor.world
                print(
                    f"sim={executor.elapsed_sim_time:7.1f}s  "
                    f"speed={executor.ego_state.velocity * 3.6:6.1f} km/h  "
                    f"s={world.last_track_state.s:7.1f} m"
                )
                next_report += 10.0
        if not executor.stopped:
            executor.stop()
            for task in executor.task_runner.tasks:
                finish = getattr(task, "finish", None)
                if callable(finish):
                    finish("timeout")
        result: dict[str, Any] = {
            "profile": profile,
            "sim_time_s": float(executor.elapsed_sim_time),
            "speed_kmh": float(executor.ego_state.velocity * 3.6),
        }
        for task in executor.task_runner.tasks:
            outputs = getattr(task, "last_outputs", None)
            if outputs:
                result.update(outputs)
        return result
    finally:
        executor.world.close()


def launch_ui(
    profile: str = DEFAULT_PROFILE,
    *,
    experiment_config: str | None = None,
    backend: str | None = None,
) -> None:
    """Launch AVLite's graphical stack visualizer with the A2RL profile active."""

    load_profile(
        profile,
        experiment_config=experiment_config,
        backend=backend,
    )
    _bootstrap_avlite_race_plot()
    from avlite.__main__ import main as avlite_main

    avlite_main([])


def _bootstrap_avlite_race_plot() -> str | None:
    """Prime AVLite's Tk visualizer for a custom race-planner strategy.

    AVLite 0.5.x selects its initial canvas from two built-in planner names
    before it reloads the selected profile.  The normal reload path supports
    custom registered planners, so a temporary built-in name is sufficient to
    construct the race canvas; the profile restores the A2RL planner before the
    executor is built.
    """

    from avlite.c20_planning.c25_global_race_planners import (
        GlobalCenterlineRacePlanner,
    )
    from avlite.c40_execution.c49_settings import ExecutionSettings

    configured_planner = ExecutionSettings.c40_global_planner
    if configured_planner == "A2RLGlobalRacePlanner":
        ExecutionSettings.c40_global_planner = GlobalCenterlineRacePlanner.__name__
    return configured_planner


def _step_executor(executor) -> None:
    from avlite.c40_execution.c49_settings import ExecutionSettings

    executor.step(
        perception_dt=ExecutionSettings.c40_perception_dt,
        control_dt=ExecutionSettings.c40_control_dt,
        replan_dt=ExecutionSettings.c40_replan_dt,
        localization_dt=ExecutionSettings.c40_localization_dt,
        sim_dt=ExecutionSettings.c40_sim_dt,
        call_replan=True,
        call_control=True,
        call_perceive=bool(ExecutionSettings.c40_perception),
        call_localize=bool(ExecutionSettings.c40_localization),
        pace_perception=True,
        pace_replan=True,
        pace_control=True,
        pace_sim=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aa stack",
        description="Run the Chrono A2RL application through AVLite.",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--config", dest="experiment_config")
    parser.add_argument("--backend", choices=["chrono", "mock"])
    parser.add_argument("--max-sim-time", type=float)
    parser.add_argument("--ui", action="store_true", help="Open AVLite's GUI.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Assemble and step once, then exit.",
    )
    args = parser.parse_args(argv)
    if args.ui:
        launch_ui(
            args.profile,
            experiment_config=args.experiment_config,
            backend=args.backend,
        )
        return
    if args.validate:
        result = validate_stack(
            args.profile,
            experiment_config=args.experiment_config,
            backend=args.backend or "mock",
        )
    else:
        result = run_stack(
            args.profile,
            experiment_config=args.experiment_config,
            backend=args.backend,
            max_sim_time=args.max_sim_time,
        )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
