#!/usr/bin/env python3
"""Run one MPC-controlled lap with ChronoDirectBackend or the mock fallback."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chrono_a2rl.chrono_interface.direct_backend import ChronoDirectBackend
from chrono_a2rl.chrono_interface.reset_manager import initial_state_from_track
from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.common.logging import get_logger
from chrono_a2rl.common.types import VehicleCommand
from chrono_a2rl.control.mpc_lateral import LateralMPCController
from chrono_a2rl.control.reference_generator import make_reference
from chrono_a2rl.control.safety_supervisor import SafetySupervisor
from chrono_a2rl.control.speed_pid import SpeedPIDController
from chrono_a2rl.evaluation.metrics import (
    MPS_TO_KMH,
    compute_metrics,
    format_lap_time,
    metrics_to_dict,
)
from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_loader import load_track_from_config


LOGGER = get_logger("run_mpc_lap")


def run(config_path: str | Path, backend_override: str | None = None) -> dict[str, Any]:
    """Run the configured closed-loop MPC rollout."""

    config = load_experiment_config(config_path)
    track = load_track_from_config(config["track"])
    speed_profile = generate_speed_profile(track, config.get("speed_profile", {}))

    simulation = config["simulation"]
    if backend_override is not None:
        simulation["backend"] = backend_override
    vehicle = config["vehicle"]
    dt = float(simulation.get("control_dt", 0.02))
    max_time = float(simulation.get("max_episode_time", 180.0))
    initial_speed = float(simulation.get("initial_speed", 3.0))
    max_offtrack_steps = int(simulation.get("max_offtrack_steps", 20))

    backend = ChronoDirectBackend(vehicle, simulation)
    initial_state = initial_state_from_track(track, s=0.0, speed=initial_speed)
    state = backend.reset(initial_state)
    episode_start_time = float(state.sim_time)
    lateral = LateralMPCController(config["controller"].get("lateral", {}), vehicle)
    speed = SpeedPIDController(config["controller"].get("speed", {}))
    supervisor = SafetySupervisor(vehicle, simulation)

    rows: list[dict[str, Any]] = []
    termination_reason = "timeout"
    previous_s = track.track_state_at_pose(state.x, state.y, state.yaw).s
    accumulated_progress = 0.0
    offtrack_streak = 0
    termination_config = config.get("termination", {})
    start_finish_s = track.wrap_s(float(termination_config.get("start_finish_s", 0.0)))
    minimum_lap_fraction = float(termination_config.get("minimum_lap_fraction", 0.95))
    stop_on_offtrack = bool(termination_config.get("stop_on_offtrack", True))
    stop_on_instability = bool(termination_config.get("stop_on_instability", True))

    max_steps = int(max_time / dt)
    for step in range(max_steps):
        track_state = track.track_state_at_pose(state.x, state.y, state.yaw)
        reference = make_reference(
            track,
            speed_profile,
            track_state,
            horizon_steps=lateral.horizon_steps,
            control_dt=dt,
        )
        steer_cmd = lateral.compute_command(state, track_state, reference, dt)
        speed_cmd = speed.compute_command(state, track_state, reference, dt)
        command = VehicleCommand(
            steering_target=steer_cmd.steering_target,
            throttle_target=speed_cmd.throttle_target,
            brake_target=speed_cmd.brake_target,
            gear_request=speed_cmd.gear_request,
            command_timestamp=state.sim_time,
            command_valid_until=state.sim_time + dt,
        )
        safe_command = supervisor.supervise(command, state, track_state, dt)
        state = backend.step(safe_command, dt)
        next_track_state = track.track_state_at_pose(state.x, state.y, state.yaw)

        previous_track_s = previous_s
        ds = next_track_state.s - previous_track_s
        if ds < -0.5 * track.length:
            ds += track.length
        elif ds > 0.5 * track.length:
            ds -= track.length
        accumulated_progress += max(0.0, ds)
        previous_s = next_track_state.s
        crossed_start_finish = track.crossed_line_forward(
            previous_track_s,
            next_track_state.s,
            line_s=start_finish_s,
        )
        lap_completed = (
            crossed_start_finish
            and accumulated_progress >= minimum_lap_fraction * track.length
        )

        if not next_track_state.on_track:
            offtrack_streak += 1
        else:
            offtrack_streak = 0

        control_saturated = supervisor.last_saturated or backend.last_control_saturated
        row = {
            "step": step,
            "sim_time": format_lap_time(state.sim_time),
            "sim_time_seconds": state.sim_time,
            "episode_time_seconds": state.sim_time - episode_start_time,
            "x": state.x,
            "y": state.y,
            "yaw": state.yaw,
            "speed_kmh": state.speed * MPS_TO_KMH,
            "yaw_rate": state.yaw_rate,
            "steering_angle": state.steering_angle,
            "throttle": state.throttle,
            "brake": state.brake,
            "s": next_track_state.s,
            "progress_s": accumulated_progress,
            "start_finish_s": start_finish_s,
            "crossed_start_finish": crossed_start_finish,
            "lateral_error": next_track_state.n,
            "heading_error": next_track_state.heading_error,
            "curvature": next_track_state.curvature,
            "distance_left_boundary": next_track_state.distance_left_boundary,
            "distance_right_boundary": next_track_state.distance_right_boundary,
            "on_track": next_track_state.on_track,
            "on_curb": next_track_state.on_curb,
            "curb_side": next_track_state.curb_side,
            "curb_penalty_weight": (
                next_track_state.curb_penalty_weight if next_track_state.on_curb else 0.0
            ),
            "target_speed_kmh": reference.target_speed * MPS_TO_KMH,
            "steering_target": safe_command.steering_target,
            "throttle_target": safe_command.throttle_target,
            "brake_target": safe_command.brake_target,
            "emergency_brake": safe_command.emergency_brake,
            "control_saturated": control_saturated,
            "supervisor_reason": supervisor.last_reason,
            "lateral_controller_mode": lateral.mode,
            "mpc_solver_status": lateral.last_solver_status,
            "mpc_nominal_steering": lateral.last_nominal_steering,
            "mpc_ancillary_correction": lateral.last_ancillary_correction,
            "mpc_tube_lateral_bound": float(lateral.last_tube_state_bound[0]),
            "mpc_tube_heading_bound": float(lateral.last_tube_state_bound[1]),
            "mpc_tube_steering_bound": float(lateral.last_tube_state_bound[2]),
            "mpc_tube_input_bound": lateral.last_tube_input_bound,
            "termination_reason": "",
        }
        if bool(config.get("logging", {}).get("include_vehicle_telemetry", True)):
            vehicle_telemetry = backend.get_telemetry()
            vehicle_telemetry.off_track = not next_track_state.on_track
            vehicle_telemetry.on_curb = next_track_state.on_curb
            row.update(vehicle_telemetry.to_dict())
        rows.append(row)

        if lap_completed:
            termination_reason = "lap_completed"
            break
        if stop_on_offtrack and offtrack_streak >= max_offtrack_steps:
            termination_reason = "off_track"
            break
        if stop_on_instability and abs(state.yaw_rate) > float(simulation.get("max_abs_yaw_rate", 2.5)) * 1.5:
            termination_reason = "instability"
            break

    if rows:
        rows[-1]["termination_reason"] = termination_reason
    metrics = compute_metrics(rows, termination_reason)
    output_paths = _save_outputs(config, rows, metrics)
    summary = metrics_to_dict(metrics)
    summary.update(output_paths)
    LOGGER.info("MPC run summary: %s", summary)
    backend.close()
    return summary


def _save_outputs(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    metrics,
) -> dict[str, str]:
    log_cfg = config.get("logging", {})
    log_dir = ROOT / str(log_cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = str(config.get("experiment", {}).get("name", "mpc_run"))
    csv_path = log_dir / f"{name}_{stamp}.csv"
    metrics_path = log_dir / f"{name}_{stamp}_metrics.yaml"
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    metrics_dict = metrics_to_dict(metrics)
    metrics_text = "\n".join(f"{key}: {value}" for key, value in metrics_dict.items()) + "\n"
    metrics_path.write_text(metrics_text, encoding="utf-8")
    return {"log_csv": str(csv_path), "metrics_yaml": str(metrics_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/mpc_yas_marina_flat.yaml",
        help="Experiment config path.",
    )
    parser.add_argument(
        "--backend",
        choices=["chrono", "mock"],
        help="Override simulation backend from the experiment config.",
    )
    args = parser.parse_args()
    summary = run(args.config, backend_override=args.backend)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
