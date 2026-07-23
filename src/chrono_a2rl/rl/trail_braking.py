"""Geometry-derived trail-braking reference for longitudinal RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chrono_a2rl.common.math_utils import clamp
from chrono_a2rl.rl.corner_progress import CornerProgressTracker, CornerSegment
from chrono_a2rl.track.speed_profile import SpeedProfile


@dataclass(frozen=True, slots=True)
class TrailBrakingReference:
    """Advisory brake target derived from the next corner geometry."""

    active: bool = False
    phase: float = 0.0
    corner_id: int = -1
    target_brake: float = 0.0
    apex_speed: float = 0.0
    distance_to_entry: float = 0.0
    distance_to_apex: float = 0.0
    required_deceleration: float = 0.0
    coast_deceleration: float = 0.0


def compute_trail_braking_reference(
    *,
    tracker: CornerProgressTracker,
    speed_profile: SpeedProfile,
    s: float,
    speed: float,
    vehicle_config: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> TrailBrakingReference:
    """Compute a moderate brake target that releases toward the apex.

    The result is an observation and reward reference, never an actuator
    override. Aerodynamic drag and rolling resistance are removed from the
    required deceleration before brake pressure is requested.
    """

    cfg = config or {}
    lookahead = max(float(cfg.get("trail_braking_lookahead_m", 350.0)), 1.0)
    reserve = max(float(cfg.get("trail_braking_reserve_m", 45.0)), 0.0)
    taper_exponent = max(float(cfg.get("trail_braking_taper_exponent", 0.80)), 0.0)
    maximum_target = clamp(
        float(cfg.get("trail_braking_max_reference", 0.70)),
        0.0,
        1.0,
    )

    segment = _reference_segment(tracker, s)
    if segment is None:
        return TrailBrakingReference()

    distance_to_entry = tracker.forward_distance(s, segment.entry_s)
    inside_corner = tracker.active_segment is segment
    if inside_corner:
        distance_to_entry = 0.0
        entry_to_apex = tracker.forward_distance(segment.entry_s, segment.apex_s)
        distance_from_entry = tracker.forward_distance(segment.entry_s, s)
        if (
            tracker.apex_passed
            or tracker.phase == "clearance"
            or distance_from_entry >= entry_to_apex
        ):
            return _inactive_reference(segment, speed_profile, phase=1.0)

    distance_to_apex = tracker.forward_distance(s, segment.apex_s)
    if distance_to_apex <= 1.0e-6 or distance_to_apex > lookahead:
        return _inactive_reference(
            segment,
            speed_profile,
            distance_to_entry=distance_to_entry,
            distance_to_apex=distance_to_apex,
            phase=1.0 if distance_to_apex <= 1.0e-6 else 0.0,
        )

    apex_speed = speed_profile.speed_at(segment.apex_s)
    speed_margin = max(float(cfg.get("trail_braking_speed_margin_fraction", 0.01)), 0.0)
    if speed <= apex_speed * (1.0 + speed_margin):
        return TrailBrakingReference(
            corner_id=segment.corner_id,
            apex_speed=apex_speed,
            distance_to_entry=distance_to_entry,
            distance_to_apex=distance_to_apex,
        )

    entry_to_apex = max(
        tracker.forward_distance(segment.entry_s, segment.apex_s),
        1.0,
    )
    braking_reference_distance = entry_to_apex if inside_corner else distance_to_apex
    available_distance = max(braking_reference_distance - reserve, 1.0)
    required_deceleration = max(
        0.0,
        (speed * speed - apex_speed * apex_speed) / (2.0 * available_distance),
    )
    drag_coefficient = max(float(vehicle_config.get("drag_coefficient", 0.0)), 0.0)
    rolling_resistance = (
        max(float(vehicle_config.get("rolling_resistance", 0.0)), 0.0)
        if speed > 0.1
        else 0.0
    )
    coast_deceleration = drag_coefficient * speed * speed + rolling_resistance
    brake_deceleration = max(
        required_deceleration - coast_deceleration,
        0.0,
    )
    maximum_deceleration = max(float(vehicle_config.get("max_decel", 16.0)), 1.0e-6)
    target_brake = clamp(
        brake_deceleration / maximum_deceleration,
        0.0,
        maximum_target,
    )

    phase = clamp(1.0 - distance_to_apex / lookahead, 0.0, 1.0)
    if inside_corner:
        remaining_fraction = clamp(distance_to_apex / entry_to_apex, 0.0, 1.0)
        target_brake *= remaining_fraction**taper_exponent

    return TrailBrakingReference(
        active=target_brake > 1.0e-6,
        phase=phase,
        corner_id=segment.corner_id,
        target_brake=target_brake,
        apex_speed=apex_speed,
        distance_to_entry=distance_to_entry,
        distance_to_apex=distance_to_apex,
        required_deceleration=required_deceleration,
        coast_deceleration=coast_deceleration,
    )


def _reference_segment(
    tracker: CornerProgressTracker,
    s: float,
) -> CornerSegment | None:
    if tracker.active_segment is not None:
        return tracker.active_segment
    return tracker.next_segment(s)


def _inactive_reference(
    segment: CornerSegment,
    speed_profile: SpeedProfile,
    *,
    distance_to_entry: float = 0.0,
    distance_to_apex: float = 0.0,
    phase: float = 0.0,
) -> TrailBrakingReference:
    return TrailBrakingReference(
        phase=phase,
        corner_id=segment.corner_id,
        apex_speed=speed_profile.speed_at(segment.apex_s),
        distance_to_entry=distance_to_entry,
        distance_to_apex=distance_to_apex,
    )
