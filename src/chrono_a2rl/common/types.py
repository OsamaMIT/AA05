"""Shared typed data containers for simulation, control, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class VehicleState:
    """Vehicle state in world coordinates plus actuator feedback."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    speed: float = 0.0
    yaw_rate: float = 0.0
    steering_angle: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    gear: int = 1
    sim_time: float = 0.0


@dataclass(slots=True)
class VehicleCommand:
    """Drive-by-wire style command sent to the backend."""

    steering_target: float = 0.0
    throttle_target: float = 0.0
    brake_target: float = 0.0
    gear_request: int = 1
    emergency_brake: bool = False
    command_timestamp: float = 0.0
    command_valid_until: float = 0.0


@dataclass(slots=True)
class TrackState:
    """Vehicle pose expressed relative to a closed-loop track."""

    s: float = 0.0
    n: float = 0.0
    heading_error: float = 0.0
    curvature: float = 0.0
    distance_left_boundary: float = 0.0
    distance_right_boundary: float = 0.0
    on_track: bool = True
    on_curb: bool = False
    curb_penalty_weight: float = 0.0
    curb_side: str = ""


@dataclass(slots=True)
class ControllerReference:
    """Reference consumed by low-level controllers."""

    target_speed: float = 0.0
    target_s: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    target_yaw: float = 0.0
    target_curvature: float = 0.0
    target_lateral_offset: float = 0.0
    current_target_yaw: float = 0.0
    current_target_curvature: float = 0.0
    curvature_preview: tuple[float, ...] = ()


@dataclass(slots=True)
class ControllerOutput:
    """Controller command with optional diagnostics."""

    command: VehicleCommand
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeMetrics:
    """Aggregate results for one closed-loop rollout."""

    lap_completed: bool = False
    lap_time: float = 0.0
    lap_time_formatted: str = "0:00.000"
    mean_speed: float = 0.0
    max_speed: float = 0.0
    lateral_error_rms: float = 0.0
    max_lateral_error: float = 0.0
    heading_error_rms: float = 0.0
    off_track_count: int = 0
    curb_sample_count: int = 0
    curb_usage_fraction: float = 0.0
    curb_penalty_total: float = 0.0
    control_saturation_count: int = 0
    mean_speed_scale: float = 0.0
    min_speed_scale: float = 0.0
    max_speed_scale: float = 0.0
    mean_target_speed_kmh: float = 0.0
    max_target_speed_kmh: float = 0.0
    profile_speed_error_rmse_kmh: float = 0.0
    profile_speed_error_mae_kmh: float = 0.0
    mean_longitudinal_action: float = 0.0
    min_longitudinal_action: float = 0.0
    max_longitudinal_action: float = 0.0
    mean_throttle: float = 0.0
    mean_brake: float = 0.0
    braking_fraction: float = 0.0
    max_validated_progress_m: float = 0.0
    frontier_progress_m: float = 0.0
    frontier_advancement_m: float = 0.0
    frontier_cleared: bool = False
    training_role: str = "evaluation"
    corner_completion_count: int = 0
    mean_corner_score: float = 0.0
    max_corner_score: float = 0.0
    mean_apex_speed_kmh: float = 0.0
    mean_exit_speed_kmh: float = 0.0
    kinetic_crash_penalty: float = 0.0
    termination_reason: str = "not_started"


@dataclass(slots=True)
class SimulationConfig:
    """Core simulation timing and runtime settings."""

    physics_dt: float = 0.001
    control_dt: float = 0.02
    max_episode_time: float = 180.0
    headless: bool = True
    backend: str = "mock"
