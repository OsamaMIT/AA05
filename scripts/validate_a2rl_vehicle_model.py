#!/usr/bin/env python3
"""Run calibration-oriented validation tests for the EAV24 approximation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chrono_a2rl.common.types import VehicleCommand, VehicleState
from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig
from chrono_a2rl.vehicle.dynamic_bicycle import DynamicBicycleModel
from run_mpc_lap import run as run_lap


def validate(*, output_dir: Path, run_laps: bool = True) -> dict:
    config = A2RLVehicleConfig.load()
    output_dir.mkdir(parents=True, exist_ok=True)
    acceleration = _acceleration_test(config)
    braking = _braking_test(config)
    cornering = _cornering_test(config)
    steering = _step_steering_test(config)
    actuator = _actuator_response_test(config)
    _plot_primitives(
        output_dir,
        acceleration,
        braking,
        cornering,
        steering,
        actuator,
    )
    results = {
        "model_warning": (
            "A2RL-style approximation; tire, aero, inertia, suspension, and "
            "actuator values require calibration."
        ),
        "acceleration": {
            "top_speed_kmh": acceleration["speed_kmh"][-1],
            "time_to_250_kmh_s": _first_time_at_speed(acceleration, 250.0),
            "target_check": 290.0 <= acceleration["speed_kmh"][-1] <= 310.0,
        },
        "braking": {
            "distance_300_to_80_m": braking["distance_m"][-1],
            "time_300_to_80_s": braking["time_s"][-1],
            "peak_decel_g": abs(min(braking["ax_mps2"])) / 9.81,
            "plausibility_check": 80.0 <= braking["distance_m"][-1] <= 300.0,
        },
        "constant_radius": {
            "peak_lateral_accel_g": max(abs(v) for v in cornering["ay_mps2"])
            / 9.81,
            "peak_tire_usage": max(cornering["tire_usage"]),
            "grip_clamp_check": max(cornering["tire_usage"]) <= 0.921,
        },
        "step_steering": {
            "initial_actual_steering_rad": steering["actual_rad"][0],
            "peak_yaw_rate_rad_s": max(abs(v) for v in steering["yaw_rate_rad_s"]),
            "delayed_response_check": steering["actual_rad"][0] == 0.0,
        },
        "actuator": {
            "target_steering_rad": actuator["target_rad"][-1],
            "actual_after_20ms_rad": actuator["actual_rad"][1],
            "rate_and_lag_check": actuator["actual_rad"][1]
            < actuator["target_rad"][1],
        },
    }
    if run_laps:
        dynamic = run_lap(
            "configs/experiments/a2rl_dynamic_vehicle_yas_marina.yaml",
            backend_override="mock",
        )
        kinematic = run_lap(
            "configs/experiments/mpc_yas_marina_flat.yaml",
            backend_override="mock",
        )
        results["full_lap_sanity"] = {
            "dynamic": _lap_summary(dynamic),
            "level0_kinematic_comparison": _lap_summary(kinematic),
            "note": (
                "A dynamic-model failure identifies controller/calibration work; "
                "it must not be hidden by relaxing tire saturation."
            ),
        }
        if dynamic.get("log_csv"):
            _plot_lap_telemetry(Path(dynamic["log_csv"]), output_dir)
    summary_path = output_dir / "validation_summary.yaml"
    summary_path.write_text(yaml.safe_dump(results, sort_keys=False), encoding="utf-8")
    results["summary_path"] = str(summary_path)
    return results


def _new_model(config: A2RLVehicleConfig) -> DynamicBicycleModel:
    return DynamicBicycleModel(config, physics_dt=0.002)


def _acceleration_test(config: A2RLVehicleConfig) -> dict[str, list[float]]:
    model = _new_model(config)
    model.reset(VehicleState(speed=5.0))
    data = {"time_s": [], "speed_kmh": [], "downforce_n": [], "drag_force_n": []}
    for _ in range(6000):
        state = model.step(VehicleCommand(throttle_target=1.0), 0.01)
        telemetry = model.get_telemetry()
        data["time_s"].append(state.sim_time)
        data["speed_kmh"].append(state.speed * 3.6)
        data["downforce_n"].append(telemetry.downforce_n)
        data["drag_force_n"].append(telemetry.drag_force_n)
    return data


def _braking_test(config: A2RLVehicleConfig) -> dict[str, list[float]]:
    model = _new_model(config)
    model.reset(VehicleState(speed=300.0 / 3.6))
    data = {"time_s": [], "distance_m": [], "speed_kmh": [], "ax_mps2": []}
    while model.get_state().speed > 80.0 / 3.6 and len(data["time_s"]) < 1500:
        state = model.step(VehicleCommand(brake_target=1.0), 0.01)
        data["time_s"].append(state.sim_time)
        data["distance_m"].append(state.x)
        data["speed_kmh"].append(state.speed * 3.6)
        data["ax_mps2"].append(model.get_telemetry().longitudinal_accel_mps2)
    return data


def _cornering_test(config: A2RLVehicleConfig) -> dict[str, list[float]]:
    model = _new_model(config)
    model.reset(VehicleState(speed=40.0))
    command = VehicleCommand(steering_target=0.08, throttle_target=0.25)
    data = {"time_s": [], "ay_mps2": [], "tire_usage": [], "yaw_rate_rad_s": []}
    for _ in range(400):
        state = model.step(command, 0.01)
        telemetry = model.get_telemetry()
        data["time_s"].append(state.sim_time)
        data["ay_mps2"].append(telemetry.lateral_accel_mps2)
        data["tire_usage"].append(telemetry.combined_slip_usage)
        data["yaw_rate_rad_s"].append(state.yaw_rate)
    return data


def _step_steering_test(config: A2RLVehicleConfig) -> dict[str, list[float]]:
    model = _new_model(config)
    model.reset(VehicleState(speed=40.0))
    data = {"time_s": [], "target_rad": [], "actual_rad": [], "yaw_rate_rad_s": []}
    command = VehicleCommand(steering_target=0.06)
    for _ in range(200):
        state = model.step(command, 0.005)
        data["time_s"].append(state.sim_time)
        data["target_rad"].append(command.steering_target)
        data["actual_rad"].append(state.steering_angle)
        data["yaw_rate_rad_s"].append(state.yaw_rate)
    return data


def _actuator_response_test(config: A2RLVehicleConfig) -> dict[str, list[float]]:
    return _step_steering_test(config)


def _first_time_at_speed(data: dict[str, list[float]], speed_kmh: float) -> float | None:
    for time_s, speed in zip(data["time_s"], data["speed_kmh"], strict=True):
        if speed >= speed_kmh:
            return float(time_s)
    return None


def _lap_summary(summary: dict) -> dict:
    return {
        "lap_completed": summary.get("lap_completed", False),
        "lap_time": summary.get("lap_time"),
        "max_speed_kmh": summary.get("max_speed_kmh"),
        "termination_reason": summary.get("termination_reason"),
        "log_csv": summary.get("log_csv"),
    }


def _plot_primitives(output_dir: Path, acceleration, braking, cornering, steering, actuator) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(acceleration["time_s"], acceleration["speed_kmh"])
    axes[0, 0].set(title="Acceleration", xlabel="time (s)", ylabel="speed (km/h)")
    axes[0, 1].plot(braking["distance_m"], braking["speed_kmh"])
    axes[0, 1].set(title="300 to 80 km/h braking", xlabel="distance (m)", ylabel="speed (km/h)")
    axes[0, 2].plot(cornering["time_s"], cornering["tire_usage"])
    axes[0, 2].set(title="Combined tire usage", xlabel="time (s)", ylabel="usage")
    axes[1, 0].plot(steering["time_s"], steering["target_rad"], label="target")
    axes[1, 0].plot(steering["time_s"], steering["actual_rad"], label="actual")
    axes[1, 0].set(title="Steering response", xlabel="time (s)", ylabel="steering (rad)")
    axes[1, 0].legend()
    axes[1, 1].plot(steering["time_s"], steering["yaw_rate_rad_s"])
    axes[1, 1].set(title="Step-steer yaw response", xlabel="time (s)", ylabel="yaw rate (rad/s)")
    axes[1, 2].plot(acceleration["speed_kmh"], acceleration["downforce_n"], label="downforce")
    axes[1, 2].plot(acceleration["speed_kmh"], acceleration["drag_force_n"], label="drag")
    axes[1, 2].set(title="Aerodynamics", xlabel="speed (km/h)", ylabel="force (N)")
    axes[1, 2].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "primitive_validation.png", dpi=140)
    plt.close(fig)


def _plot_lap_telemetry(csv_path: Path, output_dir: Path) -> None:
    data = pd.read_csv(csv_path)
    s = data["s"]
    fig, axes = plt.subplots(4, 2, figsize=(15, 14))
    axes[0, 0].plot(s, data["speed_kmh"], label="actual")
    axes[0, 0].plot(s, data["target_speed_kmh"], label="target")
    axes[0, 0].set_ylabel("speed (km/h)")
    axes[0, 0].legend()
    axes[0, 1].plot(s, data["lateral_accel_mps2"])
    axes[0, 1].set_ylabel("lateral accel (m/s^2)")
    axes[1, 0].plot(s, data["tire_usage_front"], label="front")
    axes[1, 0].plot(s, data["tire_usage_rear"], label="rear")
    axes[1, 0].set_ylabel("tire usage")
    axes[1, 0].legend()
    axes[1, 1].plot(data["speed_kmh"], data["downforce_n"])
    axes[1, 1].set(xlabel="speed (km/h)", ylabel="downforce (N)")
    axes[2, 0].plot(s, data["brake_force_n"])
    axes[2, 0].set_ylabel("brake force (N)")
    axes[2, 1].plot(s, data["steering_target_rad"], label="target")
    axes[2, 1].plot(s, data["steering_actual_rad"], label="actual")
    axes[2, 1].set_ylabel("steering (rad)")
    axes[2, 1].legend()
    axes[3, 0].plot(s, data["yaw_rate_rad_s"])
    axes[3, 0].set(xlabel="distance s (m)", ylabel="yaw rate (rad/s)")
    axes[3, 1].plot(s, data["drag_force_n"], label="drag")
    axes[3, 1].plot(s, data["drive_force_n"], label="drive")
    axes[3, 1].set(xlabel="distance s (m)", ylabel="force (N)")
    axes[3, 1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lap_vehicle_telemetry.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-lap", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT
        / "logs"
        / "vehicle_validation"
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    results = validate(output_dir=output_dir, run_laps=not args.skip_lap)
    print(yaml.safe_dump(results, sort_keys=False))


if __name__ == "__main__":
    main()
