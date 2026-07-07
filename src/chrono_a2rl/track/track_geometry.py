"""Closed-loop track geometry and Frenet projection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from chrono_a2rl.common.math_utils import wrap_angle
from chrono_a2rl.common.types import TrackState
from chrono_a2rl.track.curbs import CurbMap


_EPS = 1.0e-9


@dataclass(slots=True)
class TrackSample:
    """Interpolated sample on a closed-loop track."""

    s: float
    x: float
    y: float
    heading: float
    curvature: float
    width_left: float
    width_right: float


@dataclass(slots=True)
class FrenetProjection:
    """Closest-point projection result."""

    s: float
    n: float
    x_ref: float
    y_ref: float
    heading: float
    curvature: float
    width_left: float
    width_right: float
    distance_left_boundary: float
    distance_right_boundary: float
    on_track: bool


class TrackGeometry:
    """Closed-loop 2D track geometry.

    The centerline is stored without a duplicated final point. Segment `i`
    connects point `i` to point `(i + 1) % n`.
    """

    def __init__(
        self,
        centerline: np.ndarray,
        width_left: np.ndarray | float,
        width_right: np.ndarray | float,
        *,
        name: str = "track",
        raceline: np.ndarray | None = None,
        curbs: CurbMap | None = None,
    ) -> None:
        points = np.asarray(centerline, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 4:
            raise ValueError("centerline must be an array of shape (N, 2), N >= 4")
        if np.linalg.norm(points[0] - points[-1]) < 1.0e-6:
            points = points[:-1]

        self.name = name
        self.centerline = points
        self.raceline = None if raceline is None else np.asarray(raceline, dtype=float)
        self.curbs = curbs or CurbMap([])

        self.width_left = self._as_array(width_left, len(points), "width_left")
        self.width_right = self._as_array(width_right, len(points), "width_right")

        next_points = np.roll(points, -1, axis=0)
        self.segment_vectors = next_points - points
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        if np.any(self.segment_lengths < _EPS):
            raise ValueError("track contains zero-length segments")
        self.segment_unit = self.segment_vectors / self.segment_lengths[:, None]
        self.segment_headings = np.arctan2(self.segment_unit[:, 1], self.segment_unit[:, 0])
        self.s_nodes = np.concatenate([[0.0], np.cumsum(self.segment_lengths[:-1])])
        self.length = float(np.sum(self.segment_lengths))
        self.heading = self._compute_node_heading()
        self._heading_closed_unwrapped = np.unwrap(np.concatenate([self.heading, [self.heading[0]]]))
        self.curvature = self._compute_curvature()

    @staticmethod
    def _as_array(value: np.ndarray | float, n: int, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(n, float(arr), dtype=float)
        if arr.shape != (n,):
            raise ValueError(f"{name} must be a scalar or shape ({n},)")
        return arr

    def _compute_node_heading(self) -> np.ndarray:
        prev_points = np.roll(self.centerline, 1, axis=0)
        next_points = np.roll(self.centerline, -1, axis=0)
        tangent = next_points - prev_points
        return np.arctan2(tangent[:, 1], tangent[:, 0])

    def _compute_curvature(self) -> np.ndarray:
        curv = np.zeros(len(self.centerline), dtype=float)
        for i in range(len(self.centerline)):
            p_prev = self.centerline[i - 1]
            p = self.centerline[i]
            p_next = self.centerline[(i + 1) % len(self.centerline)]
            a = np.linalg.norm(p - p_prev)
            b = np.linalg.norm(p_next - p)
            c = np.linalg.norm(p_next - p_prev)
            denom = max(a * b * c, _EPS)
            v1 = p - p_prev
            v2 = p_next - p
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            curv[i] = float(2.0 * cross / denom)
        return curv

    def wrap_s(self, s: float) -> float:
        """Wrap arc length to [0, track_length)."""

        return float(s % self.length)

    def interpolate(self, s: float) -> TrackSample:
        """Linearly interpolate track quantities at arc length `s`."""

        s_mod = self.wrap_s(s)
        idx = int(np.searchsorted(self.s_nodes, s_mod, side="right") - 1)
        idx = max(0, min(idx, len(self.centerline) - 1))
        seg_len = self.segment_lengths[idx]
        t = (s_mod - self.s_nodes[idx]) / seg_len
        next_idx = (idx + 1) % len(self.centerline)
        point = self.centerline[idx] + t * self.segment_vectors[idx]
        curvature = (1.0 - t) * self.curvature[idx] + t * self.curvature[next_idx]
        width_left = (1.0 - t) * self.width_left[idx] + t * self.width_left[next_idx]
        width_right = (1.0 - t) * self.width_right[idx] + t * self.width_right[next_idx]
        heading = wrap_angle(
            (1.0 - t) * self._heading_closed_unwrapped[idx]
            + t * self._heading_closed_unwrapped[idx + 1]
        )
        return TrackSample(
            s=s_mod,
            x=float(point[0]),
            y=float(point[1]),
            heading=float(heading),
            curvature=float(curvature),
            width_left=float(width_left),
            width_right=float(width_right),
        )

    def project_xy(self, x: float, y: float) -> FrenetProjection:
        """Project a Cartesian point onto the closest centerline segment."""

        point = np.array([x, y], dtype=float)
        rel = point[None, :] - self.centerline
        seg_len_sq = self.segment_lengths**2
        t = np.sum(rel * self.segment_vectors, axis=1) / seg_len_sq
        t_clipped = np.clip(t, 0.0, 1.0)
        projected = self.centerline + t_clipped[:, None] * self.segment_vectors
        dist_sq = np.sum((point[None, :] - projected) ** 2, axis=1)
        idx = int(np.argmin(dist_sq))

        sample_s = self.s_nodes[idx] + t_clipped[idx] * self.segment_lengths[idx]
        sample = self.interpolate(float(sample_s))
        vector_to_point = point - np.array([sample.x, sample.y])
        tangent = np.array([math.cos(sample.heading), math.sin(sample.heading)])
        signed_n = float(tangent[0] * vector_to_point[1] - tangent[1] * vector_to_point[0])
        distance_left = sample.width_left - signed_n
        distance_right = sample.width_right + signed_n
        on_track = distance_left >= 0.0 and distance_right >= 0.0
        return FrenetProjection(
            s=sample.s,
            n=signed_n,
            x_ref=sample.x,
            y_ref=sample.y,
            heading=sample.heading,
            curvature=sample.curvature,
            width_left=sample.width_left,
            width_right=sample.width_right,
            distance_left_boundary=float(distance_left),
            distance_right_boundary=float(distance_right),
            on_track=bool(on_track),
        )

    def track_state_at_pose(self, x: float, y: float, yaw: float) -> TrackState:
        """Return a TrackState for a vehicle pose."""

        projection = self.project_xy(x, y)
        curb_contact = self.curbs.contact(
            projection.s,
            projection.n,
            projection.width_left,
            projection.width_right,
        )
        return TrackState(
            s=projection.s,
            n=projection.n,
            heading_error=wrap_angle(yaw - projection.heading),
            curvature=projection.curvature,
            distance_left_boundary=projection.distance_left_boundary,
            distance_right_boundary=projection.distance_right_boundary,
            on_track=projection.on_track,
            on_curb=curb_contact.on_curb,
            curb_penalty_weight=curb_contact.penalty_weight,
            curb_side=curb_contact.side,
        )

    def sample_arrays(self) -> dict[str, np.ndarray]:
        """Return arrays useful for plotting and processing outputs."""

        return {
            "s": self.s_nodes.copy(),
            "x": self.centerline[:, 0].copy(),
            "y": self.centerline[:, 1].copy(),
            "heading": self.heading.copy(),
            "curvature": self.curvature.copy(),
            "width_left": self.width_left.copy(),
            "width_right": self.width_right.copy(),
        }
