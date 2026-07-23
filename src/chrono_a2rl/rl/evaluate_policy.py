"""Evaluate a trained Stable-Baselines3 policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from chrono_a2rl.common.config import REPO_ROOT, load_experiment_config
from chrono_a2rl.evaluation.metrics import (
    MPS_TO_KMH,
    compute_metrics,
    format_lap_time,
    metrics_to_dict,
)
from chrono_a2rl.rl.callbacks import stable_baselines3_available
from chrono_a2rl.rl.run_manager import resolve_model_spec
from chrono_a2rl.rl.train import _make_env


def evaluate_policy(
    *,
    config_path: str | Path,
    model_path: str | Path,
    backend_override: str | None = "mock",
    deterministic: bool = True,
    output_dir: str | Path = "logs",
    max_steps: int | None = None,
    randomize_resets: bool | None = None,
) -> dict[str, Any]:
    """Run a trained policy once and save rollout logs/metrics."""

    if not stable_baselines3_available():
        raise RuntimeError(
            "Stable-Baselines3 is not installed. Install optional RL dependencies with "
            "`python3 -m pip install -e .[rl]` or `python3 -m pip install stable-baselines3`."
        )

    from stable_baselines3 import PPO

    config = load_experiment_config(config_path)
    resolved_model_path = resolve_model_spec(model_path, config["rl"]["model_dir"])
    if backend_override is not None:
        config["simulation"]["backend"] = backend_override
    if randomize_resets is not None:
        config["rl"]["eval_randomize_resets"] = randomize_resets
    config["rl"]["randomize_resets"] = bool(config["rl"].get("eval_randomize_resets", False))
    env = _make_env(config)
    model = PPO.load(str(resolved_model_path), env=env)
    obs, _ = env.reset()
    episode_start_time = float(env.state.sim_time)

    dt = float(config["simulation"].get("control_dt", 0.02))
    max_episode_time = float(config["simulation"].get("max_episode_time", 180.0))
    step_limit = max_steps or int(max_episode_time / dt)
    rows: list[dict[str, Any]] = []
    total_reward = 0.0
    termination_reason = "max_steps"

    for step in range(step_limit):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        track_state = env.track.track_state_at_pose(env.state.x, env.state.y, env.state.yaw)
        action_array = np.asarray(action, dtype=float).reshape(-1)
        row = {
            "step": step,
            "sim_time": format_lap_time(env.state.sim_time),
            "sim_time_seconds": env.state.sim_time,
            "episode_time_seconds": env.state.sim_time - episode_start_time,
            "x": env.state.x,
            "y": env.state.y,
            "yaw": env.state.yaw,
            "speed_kmh": env.state.speed * MPS_TO_KMH,
            "yaw_rate": env.state.yaw_rate,
            "steering_angle": env.state.steering_angle,
            "throttle": env.state.throttle,
            "brake": env.state.brake,
            "s": track_state.s,
            "progress_s": float(info.get("progress_s", 0.0)),
            "start_finish_s": float(info.get("start_finish_s", 0.0)),
            "crossed_start_finish": bool(
                info.get("crossed_start_finish", False)
            ),
            "validated_progress_m": float(info.get("validated_progress_m", 0.0)),
            "frontier_progress_m": float(info.get("frontier_progress_m", 0.0)),
            "frontier_advancement_m": float(info.get("frontier_advancement_m", 0.0)),
            "frontier_target_distance_m": float(
                info.get("frontier_target_distance_m", 0.0)
            ),
            "frontier_cleared": bool(info.get("frontier_cleared", False)),
            "frontier_reward": float(info.get("frontier_reward", 0.0)),
            "frontier_shortfall_penalty": float(
                info.get("frontier_shortfall_penalty", 0.0)
            ),
            "training_role": str(info.get("training_role", "evaluation")),
            "lateral_error": track_state.n,
            "raceline_error": track_state.n
            - float(info.get("racing_line_offset", 0.0)),
            "heading_error": track_state.heading_error,
            "curvature": track_state.curvature,
            "distance_left_boundary": track_state.distance_left_boundary,
            "distance_right_boundary": track_state.distance_right_boundary,
            "on_track": track_state.on_track,
            "on_curb": track_state.on_curb,
            "curb_side": track_state.curb_side,
            "curb_penalty_weight": track_state.curb_penalty_weight if track_state.on_curb else 0.0,
            "curb_usage_fraction_live": float(info.get("curb_usage_fraction", 0.0)),
            "curb_streak_time": float(info.get("curb_streak_time", 0.0)),
            "target_speed_kmh": float(info.get("target_speed_kmh", 0.0)),
            "profile_speed_error_kmh": float(
                info.get("profile_speed_error_kmh", 0.0)
            ),
            "profile_pid_throttle": float(info.get("profile_pid_throttle", 0.0)),
            "profile_pid_brake": float(info.get("profile_pid_brake", 0.0)),
            "profile_residual_pedal": float(
                info.get("profile_residual_pedal", 0.0)
            ),
            "profile_residual_guard": float(
                info.get("profile_residual_guard", 0.0)
            ),
            "profile_residual_authority": float(
                info.get("profile_residual_authority", 0.0)
            ),
            "target_lateral_offset": float(info.get("target_lateral_offset", 0.0)),
            "racing_line_offset": float(info.get("racing_line_offset", 0.0)),
            "reference_s": float(info.get("reference_s", 0.0)),
            "reference_racing_line_offset": float(info.get("reference_racing_line_offset", 0.0)),
            "strategy_lateral_offset": float(info.get("strategy_lateral_offset", 0.0)),
            "strategy_lateral_offset_delta": float(info.get("strategy_lateral_offset_delta", 0.0)),
            "apex_strength": float(info.get("apex_strength", 0.0)),
            "corner_completed": bool(info.get("corner_completed", False)),
            "corner_completion_count": int(info.get("corner_completion_count", 0)),
            "corner_completion_reward": float(info.get("corner_completion_reward", 0.0)),
            "corner_score": float(info.get("corner_score", 0.0)),
            "corner_progress": float(info.get("corner_progress", 0.0)),
            "corner_distance_completion": float(
                info.get("corner_distance_completion", 0.0)
            ),
            "corner_heading_completion": float(
                info.get("corner_heading_completion", 0.0)
            ),
            "distance_to_apex": float(info.get("distance_to_apex", 0.0)),
            "apex_passed": bool(info.get("apex_passed", False)),
            "apex_quality": float(info.get("apex_quality", 0.0)),
            "apex_speed_kmh": float(info.get("apex_speed_kmh", 0.0)),
            "exit_speed_kmh": float(info.get("exit_speed_kmh", 0.0)),
            "corner_speed_quality": float(info.get("corner_speed_quality", 0.0)),
            "speed_corner_strength": float(info.get("speed_corner_strength", 0.0)),
            "future_raceline_speed_cap_kmh": float(
                info.get("future_raceline_speed_cap_kmh", 0.0)
            ),
            "speed_demand_distance_m": float(info.get("speed_demand_distance_m", 0.0)),
            "required_braking_distance_m": float(
                info.get("required_braking_distance_m", 0.0)
            ),
            "actual_deceleration": float(info.get("actual_deceleration", 0.0)),
            "corner_overspeed_fraction": float(
                info.get("corner_overspeed_fraction", 0.0)
            ),
            "corner_controlled_braking_reward": float(
                info.get("corner_controlled_braking_reward", 0.0)
            ),
            "corner_overspeed_penalty": float(
                info.get("corner_overspeed_penalty", 0.0)
            ),
            "corner_excessive_braking_penalty": float(
                info.get("corner_excessive_braking_penalty", 0.0)
            ),
            "corner_underspeed_penalty": float(
                info.get("corner_underspeed_penalty", 0.0)
            ),
            "corner_braking_reward_applied": float(
                info.get("corner_braking_reward_applied", 0.0)
            ),
            "trail_braking_active": bool(info.get("trail_braking_active", False)),
            "trail_braking_phase": float(info.get("trail_braking_phase", 0.0)),
            "trail_braking_corner_id": int(info.get("trail_braking_corner_id", -1)),
            "trail_brake_target": float(info.get("trail_brake_target", 0.0)),
            "trail_brake_command": float(info.get("trail_brake_command", 0.0)),
            "trail_brake_applied": float(info.get("trail_brake_applied", 0.0)),
            "trail_brake_alignment_error": float(
                info.get("trail_brake_alignment_error", 0.0)
            ),
            "trail_brake_alignment_reward": float(
                info.get("trail_brake_alignment_reward", 0.0)
            ),
            "trail_brake_missing_penalty": float(
                info.get("trail_brake_missing_penalty", 0.0)
            ),
            "trail_brake_excess_reference_penalty": float(
                info.get("trail_brake_excess_reference_penalty", 0.0)
            ),
            "trail_brake_release_quality": float(
                info.get("trail_brake_release_quality", 1.0)
            ),
            "trail_brake_required_deceleration": float(
                info.get("trail_brake_required_deceleration", 0.0)
            ),
            "trail_brake_coast_deceleration": float(
                info.get("trail_brake_coast_deceleration", 0.0)
            ),
            "speed_alignment_strength": float(info.get("speed_alignment_strength", 0.0)),
            "floor_speed_scale": float(info.get("floor_speed_scale", 0.0)),
            "desired_speed_scale": float(info.get("desired_speed_scale", 0.0)),
            "raw_speed_scale": float(info.get("raw_speed_scale", action_array[0] if action_array.size > 0 else 0.0)),
            "speed_action": float(info.get("speed_action", action_array[0] if action_array.size > 0 else 0.0)),
            "action_mode": str(info.get("action_mode", "")),
            "longitudinal_action": float(
                info.get("longitudinal_action", action_array[0] if action_array.size > 0 else 0.0)
            ),
            "effective_longitudinal_action": float(
                info.get("effective_longitudinal_action", 0.0)
            ),
            "longitudinal_action_change": float(
                info.get("longitudinal_action_change", 0.0)
            ),
            "no_progress_time": float(info.get("no_progress_time", 0.0)),
            "stalled": bool(info.get("stalled", False)),
            "requested_throttle": float(info.get("requested_throttle", 0.0)),
            "requested_brake": float(info.get("requested_brake", 0.0)),
            "policy_requested_throttle": float(
                info.get("policy_requested_throttle", 0.0)
            ),
            "policy_requested_brake": float(
                info.get("policy_requested_brake", 0.0)
            ),
            "commanded_throttle": float(info.get("commanded_throttle", 0.0)),
            "commanded_brake": float(info.get("commanded_brake", 0.0)),
            "applied_throttle": float(info.get("applied_throttle", env.state.throttle)),
            "applied_brake": float(info.get("applied_brake", env.state.brake)),
            "steering_target": float(info.get("steering_target", 0.0)),
            "lateral_controller_mode": str(
                info.get("lateral_controller_mode", "")
            ),
            "mpc_solver_status": str(info.get("mpc_solver_status", "")),
            "mpc_nominal_steering": float(
                info.get("mpc_nominal_steering", 0.0)
            ),
            "mpc_ancillary_correction": float(
                info.get("mpc_ancillary_correction", 0.0)
            ),
            "mpc_tube_lateral_bound": float(
                info.get("mpc_tube_lateral_bound", 0.0)
            ),
            "mpc_tube_heading_bound": float(
                info.get("mpc_tube_heading_bound", 0.0)
            ),
            "mpc_tube_input_bound": float(
                info.get("mpc_tube_input_bound", 0.0)
            ),
            "requested_speed_scale": float(info.get("requested_speed_scale", 1.0)),
            "effective_speed_scale": float(info.get("effective_speed_scale", info.get("speed_scale", 0.0))),
            "action_0": float(action_array[0]) if action_array.size > 0 else 0.0,
            "action_1": float(action_array[1]) if action_array.size > 1 else 0.0,
            "reward": float(reward),
            "kinetic_crash_penalty": float(info.get("kinetic_crash_penalty", 0.0)),
            "terminal_reward": float(info.get("terminal_reward", 0.0)),
            "total_reward": total_reward,
            "control_saturated": env.supervisor.last_saturated or env.backend.last_control_saturated,
            "termination_reason": "",
        }
        rows.append(row)

        if terminated or truncated:
            termination_reason = _termination_reason(env, track_state, truncated)
            break

    if rows:
        rows[-1]["termination_reason"] = termination_reason

    metrics = compute_metrics(rows, termination_reason)
    summary = metrics_to_dict(metrics)
    summary["total_reward"] = total_reward
    summary["mean_reward"] = total_reward / max(1, len(rows))
    summary["model_path"] = str(resolved_model_path)
    paths = _save_outputs(config, rows, summary, output_dir)
    env.close()
    return {**summary, **paths}


def _termination_reason(env, track_state, truncated: bool) -> str:
    reason = str(getattr(env, "last_termination_reason", ""))
    if reason and reason != "running":
        return reason
    if not track_state.on_track:
        return "off_track"
    if truncated:
        return "timeout"
    return "terminated"


def _save_outputs(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    log_dir = (REPO_ROOT / output_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = str(config.get("experiment", {}).get("name", "rl_policy_eval"))
    csv_path = log_dir / f"{name}_{stamp}_eval.csv"
    metrics_path = log_dir / f"{name}_{stamp}_eval_metrics.yaml"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    metrics_path.write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    return {"log_csv": str(csv_path), "metrics_yaml": str(metrics_path)}


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for deterministic/stochastic policy evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/rl_planner_yas_marina.yaml",
        help="Planner RL experiment config.",
    )
    parser.add_argument(
        "--model",
        default="latest",
        help="Path to a trained SB3 PPO model, or 'latest'.",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "chrono"],
        default="mock",
        help="Backend override for evaluation.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions instead of deterministic mean actions.",
    )
    parser.add_argument("--max-steps", type=int, help="Optional maximum number of eval steps.")
    parser.add_argument("--output-dir", default="logs", help="Directory for eval logs and metrics.")
    parser.add_argument(
        "--randomize-resets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable randomized starts during evaluation.",
    )
    args = parser.parse_args(argv)
    summary = evaluate_policy(
        config_path=args.config,
        model_path=args.model,
        backend_override=args.backend,
        deterministic=not args.stochastic,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        randomize_resets=args.randomize_resets,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main(sys.argv[1:])
