"""Geometry-driven corner segmentation and completion tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chrono_a2rl.common.math_utils import circular_moving_average, clamp, wrap_angle
from chrono_a2rl.common.types import TrackState, VehicleState
from chrono_a2rl.track.speed_profile import SpeedProfile
from chrono_a2rl.track.track_geometry import TrackGeometry


@dataclass(frozen=True, slots=True)
class CornerSegment:
    """One closed-loop corner derived from centerline curvature."""

    corner_id: int
    entry_s: float
    apex_s: float
    exit_s: float
    length: float
    expected_heading_change: float
    turn_sign: float


@dataclass(slots=True)
class CornerUpdate:
    """Diagnostics and reward produced by one corner-tracker update."""

    phase: str = "approach"
    completed: bool = False
    failed: bool = False
    reward: float = 0.0
    score: float = 0.0
    corner_id: int = -1
    distance: float = 0.0
    distance_completion: float = 0.0
    heading_completion: float = 0.0
    distance_to_apex: float = 0.0
    apex_passed: bool = False
    apex_quality: float = 0.0
    apex_speed: float = 0.0
    exit_speed: float = 0.0
    speed_quality: float = 0.0
    in_corner_or_clearance: bool = False


def build_corner_segments(
    track: TrackGeometry,
    *,
    curvature_source: str = "centerline",
    smoothing_window: int = 11,
    curvature_threshold: float = 0.003,
    merge_gap: float = 25.0,
    minimum_length: float = 20.0,
    minimum_heading_change: float = 0.20,
) -> list[CornerSegment]:
    """Extract meaningful closed-loop corner segments from track curvature."""

    path_curvature = np.asarray(
        [track.curvature_at(float(s), source=curvature_source) for s in track.s_nodes],
        dtype=float,
    )
    curvature = circular_moving_average(np.abs(path_curvature), smoothing_window)
    mask = curvature >= max(0.0, curvature_threshold)
    if np.any(mask) and not np.all(mask):
        for gap in _circular_runs(mask, value=False):
            gap_length = float(np.sum(track.segment_lengths[gap]))
            if gap_length <= merge_gap:
                mask[gap] = True

    segments: list[CornerSegment] = []
    for indices in _circular_runs(mask, value=True):
        length = float(np.sum(track.segment_lengths[indices]))
        expected_heading = float(
            np.sum(np.abs(path_curvature[indices]) * track.segment_lengths[indices])
        )
        signed_heading = float(
            np.sum(path_curvature[indices] * track.segment_lengths[indices])
        )
        if length < minimum_length or expected_heading < minimum_heading_change:
            continue
        entry_index = int(indices[0])
        apex_index = int(indices[int(np.argmax(curvature[indices]))])
        exit_index = int((indices[-1] + 1) % len(track.centerline))
        segments.append(
            CornerSegment(
                corner_id=len(segments),
                entry_s=float(track.s_nodes[entry_index]),
                apex_s=float(track.s_nodes[apex_index]),
                exit_s=float(track.s_nodes[exit_index]),
                length=length,
                expected_heading_change=expected_heading,
                turn_sign=1.0 if signed_heading >= 0.0 else -1.0,
            )
        )
    segments.sort(key=lambda segment: segment.entry_s)
    return [
        CornerSegment(
            corner_id=index,
            entry_s=segment.entry_s,
            apex_s=segment.apex_s,
            exit_s=segment.exit_s,
            length=segment.length,
            expected_heading_change=segment.expected_heading_change,
            turn_sign=segment.turn_sign,
        )
        for index, segment in enumerate(segments)
    ]


class CornerProgressTracker:
    """Track apex passage, heading rotation, and stable corner exit."""

    def __init__(
        self,
        track: TrackGeometry,
        speed_profile: SpeedProfile,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.track = track
        self.speed_profile = speed_profile
        self.config = config or {}
        self.segments = build_corner_segments(
            track,
            curvature_source=str(self.config.get("corner_curvature_source", "centerline")),
            smoothing_window=int(self.config.get("corner_smoothing_window", 11)),
            curvature_threshold=float(self.config.get("corner_curvature_threshold", 0.003)),
            merge_gap=float(self.config.get("corner_merge_gap", 25.0)),
            minimum_length=float(self.config.get("corner_segment_min_length", 20.0)),
            minimum_heading_change=float(
                self.config.get("corner_segment_min_heading_change", 0.20)
            ),
        )
        self.active_segment: CornerSegment | None = None
        self.phase = "approach"
        self.corner_distance = 0.0
        self.clearance_distance = 0.0
        self.actual_heading_change = 0.0
        self.apex_eligible = True
        self.apex_passed = False
        self.apex_quality = 0.0
        self.apex_speed = 0.0
        self.exit_speed = 0.0
        self.previous_yaw = 0.0
        self.previous_track_heading = 0.0
        self.completion_count = 0

    def reset(self, state: VehicleState, track_state: TrackState) -> None:
        """Reset traversal state while retaining immutable corner geometry."""

        self.active_segment = None
        self.phase = "approach"
        self.corner_distance = 0.0
        self.clearance_distance = 0.0
        self.actual_heading_change = 0.0
        self.apex_eligible = True
        self.apex_passed = False
        self.apex_quality = 0.0
        self.apex_speed = 0.0
        self.exit_speed = 0.0
        self.previous_yaw = state.yaw
        self.previous_track_heading = self.track.interpolate(track_state.s).heading
        self.completion_count = 0
        segment = self.segment_at(track_state.s)
        if segment is not None:
            self._activate(segment, track_state.s)

    @property
    def in_corner_or_clearance(self) -> bool:
        return self.active_segment is not None

    @property
    def heading_completion(self) -> float:
        if self.active_segment is None:
            return 0.0
        expected = max(self.active_segment.expected_heading_change, 1.0e-6)
        return clamp(self.actual_heading_change / expected, 0.0, 1.0)

    @property
    def phase_value(self) -> float:
        if self.phase == "corner":
            return 0.5
        if self.phase == "clearance":
            return 1.0
        return 0.0

    def segment_at(self, s: float) -> CornerSegment | None:
        for segment in self.segments:
            if self._forward_distance(segment.entry_s, s) <= segment.length:
                return segment
        return None

    def next_segment(self, s: float) -> CornerSegment | None:
        if not self.segments:
            return None
        return min(
            self.segments,
            key=lambda segment: self._forward_distance(s, segment.entry_s),
        )

    def distance_to_apex(self, s: float) -> float:
        segment = self.active_segment or self.next_segment(s)
        if segment is None:
            return self.track.length
        if self.active_segment is segment and self.apex_passed:
            return -self._forward_distance(segment.apex_s, s)
        return self._forward_distance(s, segment.apex_s)

    def forward_distance(self, start_s: float, end_s: float) -> float:
        """Return closed-loop forward distance between two track positions."""

        return self._forward_distance(start_s, end_s)

    def update(
        self,
        state: VehicleState,
        track_state: TrackState,
        progress_delta: float,
    ) -> CornerUpdate:
        """Advance traversal state and return any earned completion reward."""

        if self.active_segment is None:
            segment = self.segment_at(track_state.s)
            if segment is not None:
                self._activate(segment, track_state.s)

        if self.active_segment is None:
            self._remember_pose(state, track_state)
            return self._diagnostics()

        if not track_state.on_track:
            update = self._diagnostics(failed=True, in_corner_or_clearance=True)
            self._clear_active()
            self._remember_pose(state, track_state)
            return update

        segment = self.active_segment
        self._update_heading(state, track_state, progress_delta)
        if self.phase == "corner":
            self.corner_distance += max(0.0, progress_delta)
            offset = self._forward_distance(segment.entry_s, track_state.s)
            apex_offset = self._forward_distance(segment.entry_s, segment.apex_s)
            if self.apex_eligible and not self.apex_passed and offset >= apex_offset:
                self.apex_passed = True
                self.apex_speed = state.speed
                self.apex_quality = self._line_and_boundary_quality(track_state)
            if self.corner_distance >= segment.length or offset > segment.length:
                self.phase = "clearance"
                self.exit_speed = state.speed
        else:
            self.clearance_distance += max(0.0, progress_delta)

        update = self._diagnostics()
        if self.phase == "clearance":
            clearance_required = float(self.config.get("corner_exit_clearance", 40.0))
            if self.clearance_distance >= clearance_required:
                if self._completion_gates_pass(track_state):
                    score, speed_quality = self._completion_score(track_state)
                    update = self._diagnostics(
                        completed=True,
                        reward=score,
                        score=score,
                        speed_quality=speed_quality,
                    )
                    self.completion_count += 1
                    self._clear_active()
                elif self.clearance_distance >= float(
                    self.config.get("corner_max_exit_clearance", 100.0)
                ):
                    update = self._diagnostics(failed=True)
                    self._clear_active()

        self._remember_pose(state, track_state)
        return update

    def _activate(self, segment: CornerSegment, s: float) -> None:
        self.active_segment = segment
        self.phase = "corner"
        self.corner_distance = min(self._forward_distance(segment.entry_s, s), segment.length)
        self.clearance_distance = 0.0
        self.actual_heading_change = 0.0
        apex_offset = self._forward_distance(segment.entry_s, segment.apex_s)
        self.apex_eligible = self.corner_distance < apex_offset
        self.apex_passed = False
        self.apex_quality = 0.0
        self.apex_speed = 0.0
        self.exit_speed = 0.0

    def _update_heading(
        self,
        state: VehicleState,
        track_state: TrackState,
        progress_delta: float,
    ) -> None:
        if progress_delta <= 0.0:
            return
        track_heading = self.track.interpolate(track_state.s).heading
        expected_delta = wrap_angle(track_heading - self.previous_track_heading)
        actual_delta = wrap_angle(state.yaw - self.previous_yaw)
        if self.phase == "clearance" and self.active_segment is not None:
            matched_delta = actual_delta * self.active_segment.turn_sign
            self.actual_heading_change += max(0.0, matched_delta)
        elif abs(expected_delta) > 1.0e-5:
            matched_delta = actual_delta * float(np.sign(expected_delta))
            self.actual_heading_change += max(0.0, matched_delta)

    def _completion_gates_pass(self, track_state: TrackState) -> bool:
        assert self.active_segment is not None
        distance_completion = self.corner_distance / max(self.active_segment.length, 1.0e-6)
        return bool(
            self.apex_passed
            and distance_completion >= float(self.config.get("corner_min_distance_fraction", 0.90))
            and self.heading_completion
            >= float(self.config.get("corner_min_heading_fraction", 0.80))
            and abs(track_state.heading_error)
            <= float(self.config.get("corner_exit_max_heading_error", 0.15))
            and min(
                track_state.distance_left_boundary,
                track_state.distance_right_boundary,
            )
            >= float(self.config.get("corner_exit_min_boundary_margin", 0.75))
        )

    def _completion_score(
        self,
        track_state: TrackState,
    ) -> tuple[float, float]:
        assert self.active_segment is not None
        distance_completion = clamp(
            self.corner_distance / max(self.active_segment.length, 1.0e-6),
            0.0,
            1.0,
        )
        apex_profile = max(self.speed_profile.speed_at(self.active_segment.apex_s), 1.0)
        exit_profile = max(self.speed_profile.speed_at(self.active_segment.exit_s), 1.0)
        apex_ratio = self.apex_speed / apex_profile
        exit_ratio = self.exit_speed / exit_profile
        speed_quality = clamp(0.5 * (apex_ratio + exit_ratio) / 1.10, 0.0, 1.0)
        exit_quality = self._line_and_boundary_quality(track_state)
        clean_quality = min(self.apex_quality, exit_quality)
        score = (
            float(self.config.get("corner_completion_base_bonus", 200.0))
            + float(self.config.get("corner_heading_bonus_weight", 100.0))
            * self.heading_completion
            + float(self.config.get("corner_distance_bonus_weight", 100.0))
            * distance_completion
            + float(self.config.get("corner_apex_bonus_weight", 150.0)) * clean_quality
            + float(self.config.get("corner_speed_bonus_weight", 150.0))
            * speed_quality
            * clean_quality
        )
        return score, speed_quality

    def _line_and_boundary_quality(self, track_state: TrackState) -> float:
        line_scale = max(float(self.config.get("racing_line_error_scale", 1.5)), 1.0e-6)
        line_offset = self.track.raceline_lateral_offset_at(track_state.s)
        line_quality = clamp(1.0 - abs(track_state.n - line_offset) / line_scale, 0.0, 1.0)
        margin = max(float(self.config.get("corner_quality_boundary_margin", 1.2)), 1.0e-6)
        boundary_quality = clamp(
            min(
                track_state.distance_left_boundary,
                track_state.distance_right_boundary,
            )
            / margin,
            0.0,
            1.0,
        )
        return line_quality * boundary_quality

    def _diagnostics(
        self,
        *,
        completed: bool = False,
        failed: bool = False,
        reward: float = 0.0,
        score: float = 0.0,
        speed_quality: float = 0.0,
        in_corner_or_clearance: bool | None = None,
    ) -> CornerUpdate:
        segment = self.active_segment
        distance_completion = (
            clamp(self.corner_distance / max(segment.length, 1.0e-6), 0.0, 1.0)
            if segment is not None
            else 0.0
        )
        return CornerUpdate(
            phase=self.phase,
            completed=completed,
            failed=failed,
            reward=reward,
            score=score,
            corner_id=segment.corner_id if segment is not None else -1,
            distance=self.corner_distance,
            distance_completion=distance_completion,
            heading_completion=self.heading_completion,
            distance_to_apex=(
                self.distance_to_apex(self.previous_s_for_distance)
                if segment is not None and hasattr(self, "previous_s_for_distance")
                else 0.0
            ),
            apex_passed=self.apex_passed,
            apex_quality=self.apex_quality,
            apex_speed=self.apex_speed,
            exit_speed=self.exit_speed,
            speed_quality=speed_quality,
            in_corner_or_clearance=(
                segment is not None
                if in_corner_or_clearance is None
                else in_corner_or_clearance
            ),
        )

    def _remember_pose(self, state: VehicleState, track_state: TrackState) -> None:
        self.previous_yaw = state.yaw
        self.previous_track_heading = self.track.interpolate(track_state.s).heading
        self.previous_s_for_distance = track_state.s

    def _clear_active(self) -> None:
        self.active_segment = None
        self.phase = "approach"
        self.corner_distance = 0.0
        self.clearance_distance = 0.0
        self.actual_heading_change = 0.0
        self.apex_eligible = True
        self.apex_passed = False
        self.apex_quality = 0.0
        self.apex_speed = 0.0
        self.exit_speed = 0.0

    def _forward_distance(self, start_s: float, end_s: float) -> float:
        return float((end_s - start_s) % self.track.length)


def _circular_runs(mask: np.ndarray, *, value: bool) -> list[np.ndarray]:
    """Return circular runs of a boolean value as index arrays."""

    n = int(mask.size)
    if n == 0 or not np.any(mask == value):
        return []
    if np.all(mask == value):
        return [np.arange(n, dtype=int)]
    starts = [
        index
        for index in range(n)
        if bool(mask[index]) == value and bool(mask[index - 1]) != value
    ]
    runs: list[np.ndarray] = []
    for start in starts:
        indices: list[int] = []
        index = start
        while bool(mask[index % n]) == value and len(indices) < n:
            indices.append(index % n)
            index += 1
        runs.append(np.asarray(indices, dtype=int))
    return runs
