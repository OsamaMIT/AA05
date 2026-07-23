"""Live graphical policy evaluation helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.evaluation.metrics import MPS_TO_KMH, format_lap_time
from chrono_a2rl.rl.callbacks import stable_baselines3_available
from chrono_a2rl.rl.run_manager import resolve_model_spec
from chrono_a2rl.rl.train import _make_env
from chrono_a2rl.track.track_geometry import TrackGeometry


def watch_policy(
    *,
    config_path: str | Path,
    model_path: str | Path,
    backend_override: str | None = "mock",
    deterministic: bool = True,
    realtime_factor: float = 1.0,
    max_steps: int | None = None,
    trail_points: int = 300,
    randomize_resets: bool | None = None,
    camera: str = "full",
    zoom_radius: float = 120.0,
    zoom_ahead: float = 35.0,
) -> dict[str, Any]:
    """Run a trained policy and render a live top-down track view."""

    if not stable_baselines3_available():
        raise RuntimeError(
            "Stable-Baselines3 is not installed. Install optional RL dependencies with "
            "`python3 -m pip install -e .[rl]` or `python3 -m pip install stable-baselines3`."
        )

    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO

    config = load_experiment_config(config_path)
    resolved_model_path = resolve_model_spec(model_path, config["rl"]["model_dir"])
    if backend_override is not None:
        config["simulation"]["backend"] = backend_override
    if randomize_resets is not None:
        config["rl"]["eval_randomize_resets"] = randomize_resets
    config["rl"]["randomize_resets"] = bool(config["rl"].get("eval_randomize_resets", False))
    camera = str(camera).lower()
    if camera not in {"full", "follow"}:
        raise ValueError("camera must be 'full' or 'follow'")

    env = _make_env(config)
    model = PPO.load(str(resolved_model_path), env=env)
    obs, _ = env.reset()

    dt = float(config["simulation"].get("control_dt", 0.02))
    max_episode_time = float(config["simulation"].get("max_episode_time", 180.0))
    step_limit = max_steps or int(max_episode_time / dt)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(right=0.78)
    _draw_track(
        ax,
        env.track,
        start_finish_s=float(config.get("termination", {}).get("start_finish_s", 0.0)),
    )
    car, = ax.plot([], [], "o", color="#e31a1c", markersize=7, label="car")
    heading_line, = ax.plot([], [], "-", color="#e31a1c", linewidth=2)
    trail, = ax.plot([], [], "-", color="#1f78b4", linewidth=1.5, alpha=0.8, label="trail")
    text = ax.text(
        1.03,
        0.70,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), borderaxespad=0.0)
    fig.canvas.manager.set_window_title("Chrono A2RL Policy Watch")
    plt.show(block=False)

    xs: list[float] = []
    ys: list[float] = []
    total_reward = 0.0
    termination_reason = "max_steps"

    for step in range(step_limit):
        loop_start = time.perf_counter()
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        xs.append(float(env.state.x))
        ys.append(float(env.state.y))
        if len(xs) > trail_points:
            xs = xs[-trail_points:]
            ys = ys[-trail_points:]

        _update_artists(car, heading_line, trail, text, xs, ys, env, info, step, total_reward)
        _update_camera(
            ax,
            env,
            camera=camera,
            zoom_radius=zoom_radius,
            zoom_ahead=zoom_ahead,
        )
        fig.canvas.draw_idle()
        plt.pause(0.001)

        if terminated or truncated:
            termination_reason = _termination_reason(env, truncated)
            break
        if not plt.fignum_exists(fig.number):
            termination_reason = "viewer_closed"
            break

        target_wall_dt = dt / max(realtime_factor, 1.0e-6)
        elapsed = time.perf_counter() - loop_start
        if elapsed < target_wall_dt:
            time.sleep(target_wall_dt - elapsed)

    env.close()
    plt.show(block=True)
    return {
        "termination_reason": termination_reason,
        "steps": step + 1 if "step" in locals() else 0,
        "sim_time": format_lap_time(float(getattr(env.state, "sim_time", 0.0))),
        "sim_time_seconds": float(getattr(env.state, "sim_time", 0.0)),
        "total_reward": total_reward,
        "progress_s": float(getattr(env, "progress_s", 0.0)),
        "camera": camera,
        "model_path": str(resolved_model_path),
    }


def _draw_track(ax, track: TrackGeometry, *, start_finish_s: float = 0.0) -> None:
    arrays = track.sample_arrays()
    x = arrays["x"]
    y = arrays["y"]
    heading = arrays["heading"]
    normal_x = -np.sin(heading)
    normal_y = np.cos(heading)
    left_x = x + normal_x * arrays["width_left"]
    left_y = y + normal_y * arrays["width_left"]
    right_x = x - normal_x * arrays["width_right"]
    right_y = y - normal_y * arrays["width_right"]

    ax.plot(_closed(x), _closed(y), color="#222222", linewidth=1.0, label="centerline")
    ax.plot(_closed(left_x), _closed(left_y), color="#666666", linewidth=1.0, label="track limits")
    ax.plot(_closed(right_x), _closed(right_y), color="#666666", linewidth=1.0)
    if track.raceline is not None:
        ax.plot(
            _closed(track.raceline[:, 0]),
            _closed(track.raceline[:, 1]),
            color="#33a02c",
            linewidth=1.0,
            alpha=0.8,
            label="raceline",
        )
    line = track.interpolate(start_finish_s)
    line_normal = np.array([-np.sin(line.heading), np.cos(line.heading)])
    left = np.array([line.x, line.y]) + line_normal * line.width_left
    right = np.array([line.x, line.y]) - line_normal * line.width_right
    ax.plot(
        [left[0], right[0]],
        [left[1], right[1]],
        color="#111111",
        linewidth=3.0,
        label="start / finish",
        zorder=4,
    )
    margin = 40.0
    all_x = np.concatenate([x, left_x, right_x])
    all_y = np.concatenate([y, left_y, right_y])
    ax.set_xlim(float(np.min(all_x) - margin), float(np.max(all_x) + margin))
    ax.set_ylim(float(np.min(all_y) - margin), float(np.max(all_y) + margin))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{track.name} live policy evaluation")
    ax.grid(True, linewidth=0.4, alpha=0.35)


def _update_artists(
    car,
    heading_line,
    trail,
    text,
    xs: list[float],
    ys: list[float],
    env,
    info: dict[str, Any],
    step: int,
    total_reward: float,
) -> None:
    x = float(env.state.x)
    y = float(env.state.y)
    yaw = float(env.state.yaw)
    heading_length = 12.0
    car.set_data([x], [y])
    heading_line.set_data(
        [x, x + heading_length * np.cos(yaw)],
        [y, y + heading_length * np.sin(yaw)],
    )
    trail.set_data(xs, ys)
    progress_fraction = float(info.get("progress_fraction", 0.0))
    text.set_text(
        "\n".join(
            [
                f"step: {step}",
                f"sim time: {format_lap_time(env.state.sim_time)}",
                f"speed: {env.state.speed * MPS_TO_KMH:6.1f} km/h",
                f"profile: {float(info.get('target_speed_kmh', 0.0)):6.1f} km/h",
                f"pedal: {float(info.get('longitudinal_action', 0.0)):+5.2f}",
                f"throttle: {float(info.get('applied_throttle', 0.0)):5.2f}",
                f"brake: {float(info.get('applied_brake', 0.0)):5.2f}",
                f"progress: {100.0 * progress_fraction:5.1f}%",
                f"validated: {float(info.get('validated_progress_m', 0.0)):7.1f} m",
                f"frontier: {float(info.get('frontier_progress_m', 0.0)):7.1f} m",
                f"corner: {str(info.get('corner_phase', 'approach'))}",
                f"reward: {total_reward:8.2f}",
            ]
        )
    )


def _update_camera(
    ax,
    env,
    *,
    camera: str,
    zoom_radius: float,
    zoom_ahead: float,
) -> None:
    """Update the plot viewport for full-track or car-following camera modes."""

    if camera != "follow":
        return
    radius = max(20.0, float(zoom_radius))
    x = float(env.state.x)
    y = float(env.state.y)
    yaw = float(env.state.yaw)
    center_x = x + float(zoom_ahead) * np.cos(yaw)
    center_y = y + float(zoom_ahead) * np.sin(yaw)
    ax.set_xlim(center_x - radius, center_x + radius)
    ax.set_ylim(center_y - radius, center_y + radius)


def _termination_reason(env, truncated: bool) -> str:
    reason = str(getattr(env, "last_termination_reason", ""))
    if reason and reason != "running":
        return reason
    track_state = env.track.track_state_at_pose(env.state.x, env.state.y, env.state.yaw)
    if not track_state.on_track:
        return "off_track"
    if truncated:
        return "timeout"
    return "terminated"


def _closed(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, values[:1]])


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for live graphical policy evaluation."""

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
        help="Backend override for the live evaluation.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions instead of deterministic mean actions.",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="1.0 means real time, 2.0 means twice real time, 0.5 means half speed.",
    )
    parser.add_argument("--max-steps", type=int, help="Optional maximum number of steps to render.")
    parser.add_argument(
        "--camera",
        choices=["full", "follow"],
        default="full",
        help="Use full track view or a car-following zoom camera.",
    )
    parser.add_argument(
        "--zoom-radius",
        type=float,
        default=120.0,
        help="Half-width/height of the follow camera viewport in meters.",
    )
    parser.add_argument(
        "--zoom-ahead",
        type=float,
        default=35.0,
        help="Meters to bias the follow camera ahead of the car heading.",
    )
    parser.add_argument(
        "--randomize-resets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable randomized starts during watch mode.",
    )
    args = parser.parse_args(argv)

    summary = watch_policy(
        config_path=args.config,
        model_path=args.model,
        backend_override=args.backend,
        deterministic=not args.stochastic,
        realtime_factor=args.realtime_factor,
        max_steps=args.max_steps,
        randomize_resets=args.randomize_resets,
        camera=args.camera,
        zoom_radius=args.zoom_radius,
        zoom_ahead=args.zoom_ahead,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main(sys.argv[1:])
