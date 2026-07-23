"""Closed-loop optimization of a repeatable raceline speed profile."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from chrono_a2rl.common.config import REPO_ROOT, load_experiment_config
from chrono_a2rl.common.logging import get_logger
from chrono_a2rl.evaluation.metrics import format_lap_time
from chrono_a2rl.rl.corner_progress import CornerProgressTracker
from chrono_a2rl.rl.envs.chrono_racing_planner_env import ChronoRacingPlannerEnv
from chrono_a2rl.track.speed_profile import (
    SpeedProfile,
    generate_speed_profile,
    profile_from_speed_samples,
)
from chrono_a2rl.track.track_geometry import TrackGeometry
from chrono_a2rl.track.track_loader import load_track_from_config


LOGGER = get_logger(__name__)
KMH_PER_MPS = 3.6


@dataclass(slots=True)
class ProfileRollout:
    """One deterministic zero-residual profile evaluation."""

    completed: bool
    lap_time: float
    termination_reason: str
    progress_m: float
    final_s: float
    min_boundary_margin: float
    raceline_error_rms: float
    max_raceline_error: float
    max_heading_error: float
    steering_saturation_fraction: float
    max_speed_kmh: float
    segment_stats: pd.DataFrame


@dataclass(slots=True)
class ProfileOptimizationResult:
    """Paths and headline results from one profile optimization run."""

    baseline_lap_time: float
    optimized_lap_time: float
    improvement: float
    iterations: int
    profile_path: str
    history_path: str
    segment_diagnostics_path: str
    summary_path: str


def optimize_speed_profile(
    config_path: str | Path,
    *,
    backend_override: str | None = None,
    output_dir: str | Path | None = None,
    iterations_override: int | None = None,
) -> ProfileOptimizationResult:
    """Increase locally safe profile speeds until completed-lap gains converge."""

    config = load_experiment_config(config_path)
    optimizer_config = deepcopy(config.get("profile_optimization", {}))
    if iterations_override is not None:
        optimizer_config["max_iterations"] = int(iterations_override)
    track = load_track_from_config(config["track"])
    baseline = generate_speed_profile(track, config.get("speed_profile", {}))

    segment_length = max(
        float(optimizer_config.get("segment_length_m", 100.0)),
        10.0,
    )
    segment_count = int(np.ceil(track.length / segment_length))
    speed_step = max(float(optimizer_config.get("speed_step_fraction", 0.01)), 0.001)
    max_scale = max(float(optimizer_config.get("max_speed_scale", 1.12)), 1.0)
    max_iterations = max(int(optimizer_config.get("max_iterations", 8)), 0)
    convergence_patience = max(
        int(optimizer_config.get("convergence_patience", 2)),
        1,
    )

    baseline_rollout = evaluate_profile(
        config,
        track,
        baseline,
        optimizer_config,
        backend_override=backend_override,
        segment_length=segment_length,
    )
    if not baseline_rollout.completed:
        raise RuntimeError(
            "The baseline profile did not complete a lap "
            f"({baseline_rollout.termination_reason} at "
            f"{baseline_rollout.progress_m:.1f} m). "
            "A repeatable baseline is required before optimization."
        )

    accepted_rollout = baseline_rollout
    best_profile = baseline
    best_rollout = baseline_rollout
    accepted_scales = np.ones(segment_count, dtype=float)
    best_scales = accepted_scales.copy()
    locked = np.zeros(segment_count, dtype=bool)
    history = [_history_row(0, True, baseline_rollout, 0, "baseline")]
    no_improvement = 0

    LOGGER.info(
        "Profile optimizer baseline: %s, %d segments",
        format_lap_time(baseline_rollout.lap_time),
        segment_count,
    )

    for iteration in range(1, max_iterations + 1):
        healthy = _healthy_segments(
            accepted_rollout.segment_stats,
            optimizer_config,
            segment_count,
        )
        eligible = healthy & ~locked & (accepted_scales < max_scale - 1.0e-9)
        if not np.any(eligible):
            history.append(
                _history_row(
                    iteration,
                    False,
                    accepted_rollout,
                    0,
                    "converged_no_eligible_segments",
                )
            )
            break

        proposed_scales = accepted_scales.copy()
        proposed_scales[eligible] = np.minimum(
            proposed_scales[eligible] + speed_step,
            max_scale,
        )
        candidate = _profile_from_segment_scales(
            track,
            baseline,
            proposed_scales,
            segment_length,
            config.get("speed_profile", {}),
            optimizer_config,
        )
        candidate_rollout = evaluate_profile(
            config,
            track,
            candidate,
            optimizer_config,
            backend_override=backend_override,
            segment_length=segment_length,
        )
        accepted, reason = _accept_candidate(
            accepted_rollout,
            candidate_rollout,
            optimizer_config,
        )
        history.append(
            _history_row(
                iteration,
                accepted,
                candidate_rollout,
                int(np.sum(eligible)),
                reason,
            )
        )

        if accepted:
            accepted_rollout = candidate_rollout
            accepted_scales = proposed_scales
            no_improvement = 0
            if candidate_rollout.lap_time < best_rollout.lap_time:
                best_profile = candidate
                best_rollout = candidate_rollout
                best_scales = proposed_scales.copy()
            LOGGER.info(
                "Iteration %d accepted: %s (%+.3f s), %d raised segments",
                iteration,
                format_lap_time(candidate_rollout.lap_time),
                candidate_rollout.lap_time - baseline_rollout.lap_time,
                int(np.sum(eligible)),
            )
        else:
            if reason == "no_lap_time_gain":
                locked[eligible] = True
            else:
                _lock_failure_region(
                    locked,
                    candidate_rollout,
                    accepted_rollout,
                    optimizer_config,
                    segment_length,
                )
            no_improvement += 1
            LOGGER.info(
                "Iteration %d rejected: %s (%s); locked %d/%d segments",
                iteration,
                format_lap_time(candidate_rollout.lap_time),
                reason,
                int(np.sum(locked)),
                segment_count,
            )
            if no_improvement >= convergence_patience:
                break

    safe_profile = _apply_safety_margin(
        track,
        baseline,
        best_scales,
        segment_length,
        config.get("speed_profile", {}),
        optimizer_config,
    )
    safe_rollout = evaluate_profile(
        config,
        track,
        safe_profile,
        optimizer_config,
        backend_override=backend_override,
        segment_length=segment_length,
    )
    if not safe_rollout.completed:
        LOGGER.warning(
            "Safety-margin profile failed validation; retaining best validated profile."
        )
        safe_profile = best_profile
        safe_rollout = best_rollout
    output_scales = np.divide(
        safe_profile.speed,
        np.maximum(baseline.speed, 1.0e-6),
    )

    destination = _output_directory(output_dir, optimizer_config)
    paths = _save_optimization(
        destination,
        track,
        baseline,
        safe_profile,
        output_scales,
        baseline_rollout,
        best_rollout,
        safe_rollout,
        history,
        optimizer_config,
    )
    return ProfileOptimizationResult(
        baseline_lap_time=baseline_rollout.lap_time,
        optimized_lap_time=safe_rollout.lap_time,
        improvement=baseline_rollout.lap_time - safe_rollout.lap_time,
        iterations=max(int(row["iteration"]) for row in history),
        profile_path=str(paths["profile"]),
        history_path=str(paths["history"]),
        segment_diagnostics_path=str(paths["segments"]),
        summary_path=str(paths["summary"]),
    )


def evaluate_profile(
    config: dict[str, Any],
    track: TrackGeometry,
    profile: SpeedProfile,
    optimizer_config: dict[str, Any],
    *,
    backend_override: str | None,
    segment_length: float,
) -> ProfileRollout:
    """Run one flying lap with profile PID and zero PPO pedal residual."""

    env_config = deepcopy(config)
    env_config.setdefault("speed_profile", {}).pop("profile_path", None)
    if backend_override is not None:
        env_config["simulation"]["backend"] = backend_override
    env_config.setdefault("rl", {})["randomize_resets"] = False
    env_config["rl"]["training_role"] = "evaluation"
    env_config["rl"]["eval_start_at_profile_speed"] = True

    env = ChronoRacingPlannerEnv(config=env_config)
    env.speed_profile = profile
    env.corner_tracker = CornerProgressTracker(
        track,
        profile,
        env_config.get("reward", {}),
    )
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=int(optimizer_config.get("seed", 7)))
        start_time = float(env.state.sim_time)
        max_steps = int(env.max_episode_time / env.dt)
        termination_reason = "timeout"
        for _ in range(max_steps):
            _, _, terminated, truncated, info = env.step(np.array([0.0]))
            track_state = track.track_state_at_pose(
                env.state.x,
                env.state.y,
                env.state.yaw,
            )
            racing_line_offset = track.raceline_lateral_offset_at(track_state.s)
            boundary_margin = min(
                track_state.distance_left_boundary,
                track_state.distance_right_boundary,
            )
            rows.append(
                {
                    "s": track_state.s,
                    "speed_kmh": env.state.speed * KMH_PER_MPS,
                    "target_speed_kmh": profile.speed_at(track_state.s) * KMH_PER_MPS,
                    "boundary_margin": boundary_margin,
                    "raceline_error": track_state.n - racing_line_offset,
                    "heading_error": track_state.heading_error,
                    "pid_mode": str(info.get("profile_pid_mode", "")),
                    "throttle_command": float(info.get("profile_pid_throttle", 0.0)),
                    "brake_command": float(info.get("profile_pid_brake", 0.0)),
                    "steering_saturated": abs(env.state.steering_angle)
                    >= 0.98 * float(env.vehicle.get("max_steer", 0.38)),
                }
            )
            if terminated or truncated:
                termination_reason = str(
                    info.get("termination_reason", env.last_termination_reason)
                )
                break

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError("Profile evaluation produced no simulation samples")
        segment_stats = _segment_statistics(
            frame,
            track.length,
            segment_length,
            env.dt,
        )
        return ProfileRollout(
            completed=termination_reason == "lap_completed",
            lap_time=float(env.state.sim_time - start_time),
            termination_reason=termination_reason,
            progress_m=float(env.progress_s),
            final_s=float(frame["s"].iloc[-1]),
            min_boundary_margin=float(frame["boundary_margin"].min()),
            raceline_error_rms=float(
                np.sqrt(np.mean(frame["raceline_error"].to_numpy(float) ** 2))
            ),
            max_raceline_error=float(frame["raceline_error"].abs().max()),
            max_heading_error=float(frame["heading_error"].abs().max()),
            steering_saturation_fraction=float(frame["steering_saturated"].mean()),
            max_speed_kmh=float(frame["speed_kmh"].max()),
            segment_stats=segment_stats,
        )
    finally:
        env.close()


def _segment_statistics(
    rows: pd.DataFrame,
    track_length: float,
    segment_length: float,
    dt: float,
) -> pd.DataFrame:
    segment_count = int(np.ceil(track_length / segment_length))
    working = rows.copy()
    working["segment"] = np.minimum(
        (working["s"].to_numpy(float) / segment_length).astype(int),
        segment_count - 1,
    )
    records: list[dict[str, Any]] = []
    for segment in range(segment_count):
        group = working[working["segment"] == segment]
        if group.empty:
            records.append(
                {
                    "segment": segment,
                    "s_start_m": segment * segment_length,
                    "s_end_m": min((segment + 1) * segment_length, track_length),
                    "sample_count": 0,
                    "elapsed_seconds": 0.0,
                    "min_boundary_margin_m": float("-inf"),
                    "raceline_error_rms_m": float("inf"),
                    "max_raceline_error_m": float("inf"),
                    "max_heading_error_rad": float("inf"),
                    "steering_saturation_fraction": 1.0,
                    "coast_fraction": 0.0,
                    "braking_fraction": 0.0,
                    "mean_brake_command": 0.0,
                    "max_brake_command": 0.0,
                    "mean_speed_kmh": 0.0,
                }
            )
            continue
        error = group["raceline_error"].to_numpy(float)
        records.append(
            {
                "segment": segment,
                "s_start_m": segment * segment_length,
                "s_end_m": min((segment + 1) * segment_length, track_length),
                "sample_count": len(group),
                "elapsed_seconds": len(group) * dt,
                "min_boundary_margin_m": float(group["boundary_margin"].min()),
                "raceline_error_rms_m": float(np.sqrt(np.mean(error**2))),
                "max_raceline_error_m": float(np.max(np.abs(error))),
                "max_heading_error_rad": float(group["heading_error"].abs().max()),
                "steering_saturation_fraction": float(
                    group["steering_saturated"].mean()
                ),
                "coast_fraction": float((group["pid_mode"] == "coast").mean()),
                "braking_fraction": float((group["pid_mode"] == "brake").mean()),
                "mean_brake_command": float(group["brake_command"].mean()),
                "max_brake_command": float(group["brake_command"].max()),
                "mean_speed_kmh": float(group["speed_kmh"].mean()),
            }
        )
    return pd.DataFrame(records)


def _healthy_segments(
    stats: pd.DataFrame,
    config: dict[str, Any],
    segment_count: int,
) -> np.ndarray:
    healthy = (
        (stats["sample_count"].to_numpy(int) > 0)
        & (
            stats["min_boundary_margin_m"].to_numpy(float)
            >= float(config.get("healthy_boundary_margin_m", 0.75))
        )
        & (
            stats["raceline_error_rms_m"].to_numpy(float)
            <= float(config.get("healthy_raceline_error_rms_m", 0.35))
        )
        & (
            stats["max_raceline_error_m"].to_numpy(float)
            <= float(config.get("healthy_max_raceline_error_m", 1.5))
        )
        & (
            stats["max_heading_error_rad"].to_numpy(float)
            <= float(config.get("healthy_max_heading_error_rad", 0.55))
        )
        & (
            stats["steering_saturation_fraction"].to_numpy(float)
            <= float(config.get("healthy_steering_saturation_fraction", 0.05))
        )
    )
    if healthy.shape != (segment_count,):
        raise ValueError("segment diagnostics do not match optimizer segmentation")
    return healthy


def _profile_from_segment_scales(
    track: TrackGeometry,
    baseline: SpeedProfile,
    scales: np.ndarray,
    segment_length: float,
    speed_config: dict[str, Any],
    optimizer_config: dict[str, Any],
) -> SpeedProfile:
    indices = np.minimum(
        (track.s_nodes / segment_length).astype(int),
        len(scales) - 1,
    )
    node_scales = scales[indices]
    smoothing_m = max(float(optimizer_config.get("scale_smoothing_m", 25.0)), 0.0)
    if smoothing_m > 0.0 and len(node_scales) > 2:
        mean_spacing = track.length / len(node_scales)
        half_window = int(round(0.5 * smoothing_m / max(mean_spacing, 1.0e-6)))
        if half_window > 0:
            offsets = range(-half_window, half_window + 1)
            node_scales = np.mean(
                [np.roll(node_scales, offset) for offset in offsets],
                axis=0,
            )
    return profile_from_speed_samples(
        track,
        baseline.speed * node_scales,
        speed_config,
        apply_longitudinal_limits=True,
    )


def _accept_candidate(
    accepted: ProfileRollout,
    candidate: ProfileRollout,
    config: dict[str, Any],
) -> tuple[bool, str]:
    if not candidate.completed:
        return False, candidate.termination_reason
    if candidate.min_boundary_margin < float(
        config.get("hard_min_boundary_margin_m", 0.25)
    ):
        return False, "boundary_margin"
    if candidate.raceline_error_rms > min(
        float(config.get("hard_max_raceline_error_rms_m", 0.45)),
        accepted.raceline_error_rms
        + float(config.get("allowed_raceline_error_rms_regression_m", 0.05)),
    ):
        return False, "raceline_error_rms"
    if candidate.max_raceline_error > min(
        float(config.get("hard_max_raceline_error_m", 3.0)),
        accepted.max_raceline_error
        + float(config.get("allowed_raceline_error_regression_m", 0.60)),
    ):
        return False, "raceline_error"
    if candidate.max_heading_error > min(
        float(config.get("hard_max_heading_error_rad", 0.75)),
        accepted.max_heading_error
        + float(config.get("allowed_heading_error_regression_rad", 0.05)),
    ):
        return False, "heading_error"
    if candidate.steering_saturation_fraction > min(
        float(config.get("hard_max_steering_saturation_fraction", 0.10)),
        accepted.steering_saturation_fraction
        + float(config.get("allowed_saturation_regression", 0.02)),
    ):
        return False, "steering_saturation"
    minimum_gain = float(config.get("minimum_lap_time_gain_s", 0.01))
    if candidate.lap_time > accepted.lap_time - minimum_gain:
        return False, "no_lap_time_gain"
    return True, "faster_completed_lap"


def _lock_failure_region(
    locked: np.ndarray,
    candidate: ProfileRollout,
    accepted: ProfileRollout,
    config: dict[str, Any],
    segment_length: float,
) -> None:
    radius = max(int(config.get("failure_lock_radius_segments", 2)), 0)
    if not candidate.completed:
        center = min(int(candidate.final_s / segment_length), len(locked) - 1)
    else:
        merged = candidate.segment_stats.merge(
            accepted.segment_stats,
            on="segment",
            suffixes=("_candidate", "_accepted"),
        )
        time_loss = (
            merged["elapsed_seconds_candidate"]
            - merged["elapsed_seconds_accepted"]
        )
        center = int(merged.loc[time_loss.idxmax(), "segment"])
    for offset in range(-radius, radius + 1):
        locked[(center + offset) % len(locked)] = True


def _apply_safety_margin(
    track: TrackGeometry,
    baseline: SpeedProfile,
    best_scales: np.ndarray,
    segment_length: float,
    speed_config: dict[str, Any],
    optimizer_config: dict[str, Any],
) -> SpeedProfile:
    margin = np.clip(
        float(optimizer_config.get("safety_margin_fraction", 0.02)),
        0.0,
        0.2,
    )
    safe_scales = np.maximum(1.0, best_scales * (1.0 - margin))
    return _profile_from_segment_scales(
        track,
        baseline,
        safe_scales,
        segment_length,
        speed_config,
        optimizer_config,
    )


def _history_row(
    iteration: int,
    accepted: bool,
    rollout: ProfileRollout,
    raised_segments: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "accepted": accepted,
        "lap_completed": rollout.completed,
        "lap_time": format_lap_time(rollout.lap_time),
        "lap_time_seconds": rollout.lap_time,
        "progress_m": rollout.progress_m,
        "termination_reason": rollout.termination_reason,
        "raised_segment_count": raised_segments,
        "decision": reason,
        "min_boundary_margin_m": rollout.min_boundary_margin,
        "raceline_error_rms_m": rollout.raceline_error_rms,
        "max_raceline_error_m": rollout.max_raceline_error,
        "max_heading_error_rad": rollout.max_heading_error,
        "steering_saturation_fraction": rollout.steering_saturation_fraction,
        "max_speed_kmh": rollout.max_speed_kmh,
    }


def _output_directory(
    output_dir: str | Path | None,
    optimizer_config: dict[str, Any],
) -> Path:
    if output_dir is None:
        root = REPO_ROOT / str(
            optimizer_config.get("output_dir", "artifacts/profile_optimization")
        )
        output = root / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    else:
        output = Path(output_dir)
        if not output.is_absolute():
            output = REPO_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    return output.resolve()


def _save_optimization(
    output_dir: Path,
    track: TrackGeometry,
    baseline: SpeedProfile,
    optimized: SpeedProfile,
    scales: np.ndarray,
    baseline_rollout: ProfileRollout,
    best_rollout: ProfileRollout,
    output_rollout: ProfileRollout,
    history: list[dict[str, Any]],
    optimizer_config: dict[str, Any],
) -> dict[str, Path]:
    profile_path = output_dir / "optimized_speed_profile.csv"
    history_path = output_dir / "iteration_history.csv"
    segments_path = output_dir / "segment_diagnostics.csv"
    summary_path = output_dir / "optimization_summary.yaml"
    pd.DataFrame(
        {
            "s_m": track.s_nodes,
            "base_speed_kmh": baseline.speed * KMH_PER_MPS,
            "speed_kmh": optimized.speed * KMH_PER_MPS,
            "scale": scales,
        }
    ).to_csv(profile_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    segment_diagnostics = output_rollout.segment_stats.copy()
    baseline_stats = baseline_rollout.segment_stats
    comparison_columns = (
        "elapsed_seconds",
        "mean_speed_kmh",
        "coast_fraction",
        "braking_fraction",
        "mean_brake_command",
        "max_brake_command",
    )
    if len(baseline_stats) == len(segment_diagnostics):
        for column in comparison_columns:
            if column in baseline_stats:
                segment_diagnostics[f"baseline_{column}"] = baseline_stats[
                    column
                ].to_numpy()
    segment_diagnostics.to_csv(segments_path, index=False)
    summary = {
        "baseline_lap_time": format_lap_time(baseline_rollout.lap_time),
        "baseline_lap_time_seconds": baseline_rollout.lap_time,
        "best_validated_lap_time": format_lap_time(best_rollout.lap_time),
        "best_validated_lap_time_seconds": best_rollout.lap_time,
        "output_lap_time": format_lap_time(output_rollout.lap_time),
        "output_lap_time_seconds": output_rollout.lap_time,
        "output_improvement_seconds": (
            baseline_rollout.lap_time - output_rollout.lap_time
        ),
        "safety_margin_fraction": float(
            optimizer_config.get("safety_margin_fraction", 0.02)
        ),
        "lap_completed": output_rollout.completed,
        "minimum_boundary_margin_m": output_rollout.min_boundary_margin,
        "raceline_error_rms_m": output_rollout.raceline_error_rms,
        "maximum_speed_kmh": output_rollout.max_speed_kmh,
        "profile_path": str(profile_path),
    }
    summary_path.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "profile": profile_path,
        "history": history_path,
        "segments": segments_path,
        "summary": summary_path,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for `aa optimize`."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/rl_planner_yas_marina.yaml",
    )
    parser.add_argument(
        "--backend",
        choices=["chrono", "mock"],
        default="chrono",
    )
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    result = optimize_speed_profile(
        args.config,
        backend_override=args.backend,
        output_dir=args.output_dir,
        iterations_override=args.iterations,
    )
    print(f"baseline_lap_time: {format_lap_time(result.baseline_lap_time)}")
    print(f"optimized_lap_time: {format_lap_time(result.optimized_lap_time)}")
    print(f"improvement_seconds: {result.improvement:.3f}")
    print(f"iterations: {result.iterations}")
    print(f"profile_path: {result.profile_path}")
    print(f"history_path: {result.history_path}")
    print(f"segment_diagnostics_path: {result.segment_diagnostics_path}")
    print(f"summary_path: {result.summary_path}")
