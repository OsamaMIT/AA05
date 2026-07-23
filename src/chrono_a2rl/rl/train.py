"""Stable-Baselines3 training entrypoint for racing policies."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.rl.callbacks import ProgressFrontierCallback, stable_baselines3_available
from chrono_a2rl.rl.envs.chrono_racing_env import ChronoRacingEnv
from chrono_a2rl.rl.envs.chrono_racing_planner_env import ChronoRacingPlannerEnv
from chrono_a2rl.rl.frontier import (
    ProgressFrontierState,
    assign_training_role,
    frontier_sidecar_path,
    load_frontier_state,
)
from chrono_a2rl.rl.run_manager import (
    create_run_directory,
    resolve_latest_run,
    write_run_manifest,
)


def train(
    config_path: str | Path,
    *,
    auto_evaluate: bool | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train PPO policy and optionally evaluate the saved final model."""

    if not stable_baselines3_available():
        raise RuntimeError(
            "Stable-Baselines3 is not installed. Install optional RL dependencies with "
            "`python3 -m pip install -e .[rl]` or `python3 -m pip install stable-baselines3`."
        )

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    config = load_experiment_config(config_path)
    if overrides:
        _apply_training_overrides(config, overrides)
    rl_cfg = config.get("rl", {})
    n_envs = int(rl_cfg.get("n_envs", 1))
    model_root = Path(str(rl_cfg.get("model_dir", "models/ppo_speed_policy")))
    model_root.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = str(rl_cfg.get("checkpoint_prefix", "ppo_speed_policy"))
    frontier_enabled = bool(rl_cfg.get("frontier_enabled", False)) and str(
        rl_cfg.get("env_type", "speed")
    ).lower() == "planner"
    resume_from = rl_cfg.get("resume_from")
    resume_path: Path | None = None
    frontier_callback: ProgressFrontierCallback | None = None
    run_id = ""
    if frontier_enabled:
        resume_latest = str(resume_from).strip().lower() == "latest" if resume_from else False
        if resume_latest:
            run_dir = resolve_latest_run(model_root)
            resume_path = _find_latest_checkpoint(run_dir, checkpoint_prefix)
            frontier_state = _load_matching_frontier_state(
                resume_path,
                run_dir,
                initial_progress=float(rl_cfg.get("frontier_initial_progress_m", 350.0)),
            )
            run_id = frontier_state.run_id or run_dir.name
        else:
            run_id, run_dir = create_run_directory(
                model_root,
                seed=int(rl_cfg.get("seed", 0)),
            )
            frontier_state = ProgressFrontierState(
                run_id=run_id,
                frontier_progress_m=float(rl_cfg.get("frontier_initial_progress_m", 350.0)),
            )
            if resume_from:
                resume_path = resolve_resume_checkpoint(
                    resume_from=resume_from,
                    model_dir=model_root,
                    checkpoint_prefix=checkpoint_prefix,
                )
        config["rl"]["frontier_initial_progress_m"] = frontier_state.frontier_progress_m
        write_run_manifest(run_dir, run_id=run_id, config=config)
    else:
        run_dir = model_root

    env = _make_training_env(config)
    if frontier_enabled:
        frontier_callback = ProgressFrontierCallback(
            state=frontier_state,
            run_dir=run_dir,
            checkpoint_prefix=checkpoint_prefix,
            checkpoint_interval_steps=int(rl_cfg.get("checkpoint_interval_steps", 10000)),
            maximum_advance_m=float(rl_cfg.get("frontier_max_advance_m", 150.0)),
            track_length=float(env.env_method("get_track_length")[0]),
        )
        training_callback = frontier_callback
    else:
        training_callback = CheckpointCallback(
            save_freq=max(
                1,
                int(rl_cfg.get("checkpoint_interval_steps", 10000)) // max(1, n_envs),
            ),
            save_path=str(run_dir),
            name_prefix=checkpoint_prefix,
        )
    policy_kwargs: dict[str, object] = {}
    if "policy_log_std_init" in rl_cfg:
        policy_kwargs["log_std_init"] = float(rl_cfg["policy_log_std_init"])
    if "policy_net_arch" in rl_cfg:
        policy_kwargs["net_arch"] = list(rl_cfg["policy_net_arch"])

    if resume_from and resume_path is None:
        resume_path = resolve_resume_checkpoint(
            resume_from=resume_from,
            model_dir=model_root,
            checkpoint_prefix=checkpoint_prefix,
        )
    if resume_path is not None:
        print(f"Resuming PPO training from {resume_path}")
        model = PPO.load(
            str(resume_path),
            env=env,
            tensorboard_log=str(rl_cfg.get("tensorboard_log", "logs/tensorboard")),
            verbose=1,
            **_ppo_resume_kwargs(rl_cfg),
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            **_ppo_resume_kwargs(rl_cfg),
            policy_kwargs=policy_kwargs or None,
            tensorboard_log=str(rl_cfg.get("tensorboard_log", "logs/tensorboard")),
            verbose=1,
        )
    requested_timesteps = int(rl_cfg.get("total_timesteps", 100000))
    if resume_path is not None:
        print(
            "Continuing PPO for "
            f"{requested_timesteps:,} additional env steps "
            f"from {model.num_timesteps:,} to {model.num_timesteps + requested_timesteps:,}."
        )
    else:
        print(f"Training PPO for {requested_timesteps:,} env steps from scratch.")
    model.learn(
        total_timesteps=requested_timesteps,
        callback=training_callback,
        reset_num_timesteps=resume_path is None,
    )
    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))
    saved_model_path = final_model_path.with_suffix(".zip")
    if frontier_callback is not None:
        frontier_callback.save_final_state(saved_model_path)
    env.close()

    result: dict[str, Any] = {
        "model_path": str(saved_model_path),
        "run_dir": str(run_dir),
    }
    if frontier_callback is not None:
        result["frontier_progress_m"] = frontier_callback.state.frontier_progress_m
    should_eval = bool(rl_cfg.get("auto_evaluate_after_training", True))
    if auto_evaluate is not None:
        should_eval = auto_evaluate
    if should_eval:
        from chrono_a2rl.rl.evaluate_policy import evaluate_policy

        eval_summary = evaluate_policy(
            config_path=config_path,
            model_path=saved_model_path,
            backend_override=rl_cfg.get("eval_backend", "mock"),
            deterministic=bool(rl_cfg.get("eval_deterministic", True)),
            output_dir=str(rl_cfg.get("eval_output_dir", "logs")),
            max_steps=(
                int(rl_cfg["eval_max_steps"])
                if "eval_max_steps" in rl_cfg and rl_cfg["eval_max_steps"] is not None
                else None
            ),
        )
        result["evaluation"] = eval_summary
        _print_training_eval_summary(eval_summary)
    return result


def resolve_resume_checkpoint(
    resume_from: str | Path,
    model_dir: Path,
    checkpoint_prefix: str,
) -> Path:
    """Resolve a resume token into a concrete SB3 checkpoint path."""

    token = str(resume_from).strip()
    if token.lower() == "latest":
        return _find_latest_checkpoint(model_dir=model_dir, checkpoint_prefix=checkpoint_prefix)

    path = Path(token).expanduser()
    if not path.suffix and path.with_suffix(".zip").exists():
        path = path.with_suffix(".zip")
    if not path.exists():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {path}. Use `--resume latest` or pass a valid .zip path."
        )
    return path


def _load_matching_frontier_state(
    model_path: Path,
    run_dir: Path,
    *,
    initial_progress: float,
) -> ProgressFrontierState:
    """Restore the frontier paired with a model, falling back to run state."""

    paired_state = frontier_sidecar_path(model_path)
    run_state = run_dir / "frontier_state.yaml"
    if paired_state.exists():
        return load_frontier_state(paired_state)
    if run_state.exists():
        print(
            f"Frontier sidecar missing for {model_path.name}; using {run_state.name}."
        )
        return load_frontier_state(run_state)
    print("No saved frontier state found; restoring the configured initial frontier.")
    return ProgressFrontierState(
        run_id=run_dir.name,
        frontier_progress_m=initial_progress,
    )


def _find_latest_checkpoint(model_dir: Path, checkpoint_prefix: str) -> Path:
    """Find the newest final model or checkpoint for the configured policy."""

    candidates: list[Path] = []
    final_model = model_dir / "final_model.zip"
    if final_model.exists():
        candidates.append(final_model)
    candidates.extend(model_dir.glob(f"{checkpoint_prefix}_*_steps.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {model_dir}. Train once first, or pass `--resume PATH`."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _ppo_resume_kwargs(rl_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build PPO kwargs used for fresh training and resume-time overrides."""

    return {
        "n_steps": int(rl_cfg.get("n_steps", 1024)),
        "batch_size": int(rl_cfg.get("batch_size", 256)),
        "learning_rate": float(rl_cfg.get("learning_rate", 3.0e-4)),
        "gamma": float(rl_cfg.get("gamma", 0.99)),
        "gae_lambda": float(rl_cfg.get("gae_lambda", 0.95)),
        "clip_range": float(rl_cfg.get("clip_range", 0.2)),
        "ent_coef": float(rl_cfg.get("ent_coef", 0.0)),
        "vf_coef": float(rl_cfg.get("vf_coef", 0.5)),
        "max_grad_norm": float(rl_cfg.get("max_grad_norm", 0.5)),
        "use_sde": bool(rl_cfg.get("use_sde", False)),
        "sde_sample_freq": int(rl_cfg.get("sde_sample_freq", -1)),
        "target_kl": (
            float(rl_cfg["target_kl"])
            if "target_kl" in rl_cfg and rl_cfg["target_kl"] is not None
            else None
        ),
    }


def _make_env(config):
    env_type = str(config.get("rl", {}).get("env_type", "speed")).lower()
    if env_type == "planner":
        return ChronoRacingPlannerEnv(config=config)
    return ChronoRacingEnv(config=config)


def _make_training_env(config: dict[str, Any]):
    rl_cfg = config.get("rl", {})
    n_envs = int(rl_cfg.get("n_envs", 1))
    if n_envs <= 1:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

        return VecMonitor(DummyVecEnv([_make_env_factory(config, 0)]))

    vec_env_type = str(rl_cfg.get("vec_env_type", "subproc")).lower()
    if vec_env_type == "dummy":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

        return VecMonitor(DummyVecEnv([_make_env_factory(config, i) for i in range(n_envs)]))

    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    start_method = str(rl_cfg.get("subproc_start_method", "fork"))
    return VecMonitor(
        SubprocVecEnv(
            [_make_env_factory(config, i) for i in range(n_envs)],
            start_method=start_method,
        )
    )


def _make_env_factory(base_config: dict[str, Any], env_index: int):
    def _init():
        config = deepcopy(base_config)
        rl_cfg = config.get("rl", {})
        n_envs = int(rl_cfg.get("n_envs", 1))
        if bool(rl_cfg.get("frontier_enabled", False)) and str(
            rl_cfg.get("env_type", "speed")
        ).lower() == "planner":
            rl_cfg["training_role"] = assign_training_role(env_index, n_envs, rl_cfg)
            rl_cfg["env_index"] = env_index
        train_backend = rl_cfg.get("train_backend")
        if train_backend:
            config["simulation"]["backend"] = str(train_backend)
        env = _make_env(deepcopy(config))
        seed = int(rl_cfg.get("seed", 0)) + env_index
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset(seed=seed)
        return env

    return _init


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the speed-only policy."""

    _main_with_default_config(
        argv,
        default_config="configs/experiments/rl_speed_policy_yas_marina.yaml",
    )


def main_speed(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the speed-only policy."""

    main(argv)


def main_planner(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the longitudinal pedal policy."""

    _main_with_default_config(
        argv,
        default_config="configs/experiments/rl_planner_yas_marina.yaml",
    )


def _main_with_default_config(argv: list[str] | None, *, default_config: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=default_config,
        help="RL experiment config path.",
    )
    _add_training_override_args(parser)
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip automatic evaluation after training finishes.",
    )
    args = parser.parse_args(argv)
    train(
        args.config,
        auto_evaluate=not args.no_eval,
        overrides=_training_overrides_from_args(args),
    )


def _add_training_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--total-timesteps", type=int, help="Override PPO total_timesteps.")
    parser.add_argument("--n-envs", type=int, help="Number of parallel training environments.")
    parser.add_argument("--vec-env-type", choices=["subproc", "dummy"], help="Vector env backend.")
    parser.add_argument(
        "--subproc-start-method",
        choices=["fork", "spawn", "forkserver"],
        help="Multiprocessing start method for SubprocVecEnv.",
    )
    parser.add_argument("--seed", type=int, help="Base random seed for parallel envs.")
    parser.add_argument(
        "--backend",
        "--train-backend",
        dest="train_backend",
        choices=["mock", "chrono"],
        help="Backend used during training.",
    )
    parser.add_argument("--eval-backend", choices=["mock", "chrono"], help="Backend for auto eval.")
    parser.add_argument("--model-dir", help="Directory where models/checkpoints are saved.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        dest="resume_from",
        metavar="latest|PATH",
        help="Resume PPO training from latest checkpoint or an explicit SB3 .zip path.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_const",
        const="latest",
        dest="resume_from",
        help="Resume the latest run, including its saved progress frontier.",
    )
    parser.add_argument(
        "--resume-from",
        dest="resume_from",
        metavar="PATH",
        help="Resume PPO training from an explicit SB3 .zip checkpoint.",
    )
    parser.add_argument("--checkpoint-interval-steps", type=int, help="Checkpoint interval in env steps.")
    parser.add_argument("--n-steps", type=int, help="PPO rollout length per environment.")
    parser.add_argument("--batch-size", type=int, help="PPO minibatch size.")
    parser.add_argument("--learning-rate", type=float, help="PPO learning rate.")
    parser.add_argument("--ent-coef", type=float, help="PPO entropy coefficient.")
    parser.add_argument("--clip-range", type=float, help="PPO clip range.")
    parser.add_argument("--gamma", type=float, help="PPO discount factor.")
    parser.add_argument("--gae-lambda", type=float, help="PPO GAE lambda.")
    parser.add_argument(
        "--randomize-resets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable randomized episode starts for training.",
    )
    parser.add_argument("--reset-s-mode", choices=["random", "fixed"], help="Episode start s mode.")
    parser.add_argument("--reset-s", type=float, help="Fixed reset s-coordinate when reset-s-mode=fixed.")
    parser.add_argument("--reset-s-fraction-min", type=float, help="Minimum random start fraction.")
    parser.add_argument("--reset-s-fraction-max", type=float, help="Maximum random start fraction.")
    parser.add_argument(
        "--reset-speed-mode",
        choices=["range", "profile"],
        help="Randomized reset speed source.",
    )
    parser.add_argument("--reset-speed-min", type=float, help="Minimum randomized reset speed.")
    parser.add_argument("--reset-speed-max", type=float, help="Maximum randomized reset speed.")
    parser.add_argument(
        "--reset-speed-scale-min",
        type=float,
        help="Minimum local profile speed scale for profile reset speed mode.",
    )
    parser.add_argument(
        "--reset-speed-scale-max",
        type=float,
        help="Maximum local profile speed scale for profile reset speed mode.",
    )
    parser.add_argument("--reset-lateral-offset-max", type=float, help="Maximum reset lateral offset.")
    parser.add_argument("--reset-heading-error-max", type=float, help="Maximum reset heading error.")
    parser.add_argument(
        "--longitudinal-action-deadband",
        type=float,
        help="Signed-pedal deadband around coast, in normalized action units.",
    )
    parser.add_argument(
        "--longitudinal-action-rise-rate",
        type=float,
        help="Maximum pedal-action movement toward throttle per second.",
    )
    parser.add_argument(
        "--longitudinal-action-fall-rate",
        type=float,
        help="Maximum pedal-action movement toward braking per second.",
    )
    parser.add_argument(
        "--profile-speed-tracking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable authoritative profile-speed PID tracking.",
    )
    parser.add_argument(
        "--profile-speed-residual-authority",
        type=float,
        help="Maximum PPO pedal residual as a fraction of full pedal.",
    )
    parser.add_argument(
        "--profile-speed-residual-error-guard-mps",
        type=float,
        help="Speed error where PPO residual authority fades to zero.",
    )
    parser.add_argument(
        "--vehicle-max-accel",
        type=float,
        help="Override gross kinematic tractive acceleration in m/s^2.",
    )
    parser.add_argument(
        "--use-sde",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable generalized state-dependent exploration.",
    )
    parser.add_argument(
        "--sde-sample-freq",
        type=int,
        help="Number of control steps between gSDE noise resampling.",
    )
    parser.add_argument(
        "--max-lateral-accel",
        type=float,
        help="Override raceline speed-profile lateral acceleration in m/s^2.",
    )
    parser.add_argument("--max-episode-time", type=float, help="Maximum episode time in seconds.")
    parser.add_argument("--initial-speed", type=float, help="Fixed initial speed when resets are not randomized.")
    parser.add_argument("--eval-max-steps", type=int, help="Maximum steps for automatic post-training eval.")
    parser.add_argument(
        "--eval-randomize-resets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable randomized starts during evaluation.",
    )


def _training_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "total_timesteps",
        "n_envs",
        "vec_env_type",
        "subproc_start_method",
        "seed",
        "train_backend",
        "eval_backend",
        "model_dir",
        "resume_from",
        "checkpoint_interval_steps",
        "n_steps",
        "batch_size",
        "learning_rate",
        "ent_coef",
        "clip_range",
        "gamma",
        "gae_lambda",
        "randomize_resets",
        "reset_s_mode",
        "reset_s",
        "reset_s_fraction_min",
        "reset_s_fraction_max",
        "reset_speed_mode",
        "reset_speed_min",
        "reset_speed_max",
        "reset_speed_scale_min",
        "reset_speed_scale_max",
        "reset_lateral_offset_max",
        "reset_heading_error_max",
        "longitudinal_action_deadband",
        "longitudinal_action_rise_rate",
        "longitudinal_action_fall_rate",
        "profile_speed_tracking",
        "profile_speed_residual_authority",
        "profile_speed_residual_error_guard_mps",
        "vehicle_max_accel",
        "use_sde",
        "sde_sample_freq",
        "max_lateral_accel",
        "max_episode_time",
        "initial_speed",
        "eval_max_steps",
        "eval_randomize_resets",
    )
    return {
        key: getattr(args, key)
        for key in keys
        if hasattr(args, key) and getattr(args, key) is not None
    }


def _apply_training_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    rl = config.setdefault("rl", {})
    simulation = config.setdefault("simulation", {})
    for key, value in overrides.items():
        if key in {"max_episode_time", "initial_speed"}:
            simulation[key] = value
        elif key == "max_lateral_accel":
            config.setdefault("speed_profile", {})[key] = value
        elif key == "vehicle_max_accel":
            config.setdefault("vehicle", {})["max_accel"] = value
        elif key == "profile_speed_tracking":
            rl["profile_speed_tracking_enabled"] = value
        else:
            rl[key] = value


def _print_training_eval_summary(summary: dict[str, Any]) -> None:
    print("Automatic post-training evaluation:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main(sys.argv[1:])
