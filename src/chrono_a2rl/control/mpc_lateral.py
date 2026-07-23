"""First lateral MPC controller with geometric and proportional fallbacks."""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np
from scipy import sparse

from chrono_a2rl.common.logging import get_logger
from chrono_a2rl.common.math_utils import clamp, wrap_angle
from chrono_a2rl.common.types import ControllerReference, TrackState, VehicleCommand, VehicleState
from chrono_a2rl.control.controller_interface import Controller
from chrono_a2rl.control.tube_mpc import TubeMPCSolver


LOGGER = get_logger(__name__)


class LateralMPCController(Controller):
    """Robust Tube MPC with a geometric solver-failure fallback.

    If OSQP is unavailable, the controller uses deterministic pure-pursuit
    tracking of the path preview point, with a proportional fallback when no
    preview point is available.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        vehicle_config: dict[str, Any] | None = None,
    ) -> None:
        cfg = config or {}
        vehicle = vehicle_config or {}
        self.mode = str(cfg.get("mode", "linear")).lower()
        self.tube_enabled = bool(
            cfg.get("tube_enabled", self.mode in {"tube", "full_tube"})
        )
        self.horizon_steps = int(cfg.get("horizon_steps", 15))
        self.dt = float(cfg.get("dt", 0.02))
        self.q_lat = float(cfg.get("weight_lateral_error", 8.0))
        self.q_head = float(cfg.get("weight_heading_error", 2.5))
        self.r_steer = float(cfg.get("weight_steering", 0.05))
        self.r_rate = float(cfg.get("weight_steering_rate", 0.6))
        self.k_lat = float(cfg.get("fallback_lateral_gain", 0.18))
        self.k_head = float(cfg.get("fallback_heading_gain", 1.15))
        self.fallback_method = str(cfg.get("fallback_method", "pure_pursuit")).lower()
        self.fallback_min_preview_distance = float(
            cfg.get("fallback_min_preview_distance", 2.0)
        )
        self.wheelbase = float(vehicle.get("wheelbase", 2.97))
        self.max_steer = float(vehicle.get("max_steer", 0.45))
        self.max_steer_rate = float(vehicle.get("max_steer_rate", 1.2))
        self.osqp_requested = bool(cfg.get("use_osqp", True))
        self.osqp_available = importlib.util.find_spec("osqp") is not None
        self.use_osqp = self.osqp_requested and self.osqp_available
        self.full_tube_enabled = self.mode == "full_tube"
        self.tube_radius_base = float(cfg.get("tube_radius_base", 0.25))
        self.tube_radius_speed_gain = float(cfg.get("tube_radius_speed_gain", 0.006))
        self.tube_radius_max = float(cfg.get("tube_radius_max", 1.25))
        self.tube_boundary_margin = float(cfg.get("tube_boundary_margin", 0.25))
        self.tube_soft_boundary_gain = float(cfg.get("tube_soft_boundary_gain", 0.18))
        self.last_tube_radius = 0.0
        self.last_tightened_left = math.inf
        self.last_tightened_right = math.inf
        self._previous_steer = 0.0
        self._warned_fallback = False
        self._warned_solver_failure = False
        self.last_solver_status = "not_started"
        self.last_nominal_steering = 0.0
        self.last_ancillary_correction = 0.0
        self.last_tube_state_bound = np.zeros(3, dtype=float)
        self.last_tube_input_bound = 0.0
        self.tube_solver = (
            TubeMPCSolver(
                cfg,
                wheelbase=self.wheelbase,
                max_steer=self.max_steer,
                max_steer_rate=self.max_steer_rate,
            )
            if self.full_tube_enabled and self.use_osqp
            else None
        )

    def reset(self) -> None:
        self._previous_steer = 0.0
        self.last_solver_status = "not_started"
        self.last_nominal_steering = 0.0
        self.last_ancillary_correction = 0.0
        if self.tube_solver is not None:
            self.tube_solver.reset()

    def compute_command(
        self,
        state: VehicleState,
        track_state: TrackState,
        reference: ControllerReference,
        dt: float,
    ) -> VehicleCommand:
        if self.mode == "pure_pursuit":
            self.last_solver_status = "pure_pursuit"
            steering = self._compute_fallback(state, track_state, reference)
        elif self.full_tube_enabled and self.tube_solver is not None:
            try:
                steering = self._compute_full_tube(
                    state,
                    track_state,
                    reference,
                    dt,
                )
            except RuntimeError as exc:
                self.last_solver_status = f"fallback: {exc}"
                if not self._warned_solver_failure:
                    LOGGER.warning(
                        "Full Tube MPC failed (%s). Using %s fallback.",
                        exc,
                        self.fallback_method.replace("_", "-"),
                    )
                    self._warned_solver_failure = True
                steering = self._compute_fallback(state, track_state, reference)
        elif self.use_osqp:
            steering = self._compute_osqp(state, track_state, reference, dt)
        else:
            if not self._warned_fallback:
                fallback_name = (
                    f"tube-aware {self.fallback_method.replace('_', '-')} fallback"
                    if self.tube_enabled
                    else f"{self.fallback_method.replace('_', '-')} lateral fallback"
                )
                if self.osqp_requested and not self.osqp_available:
                    LOGGER.warning("OSQP is not installed. Using %s.", fallback_name)
                else:
                    LOGGER.warning(
                        "OSQP lateral MPC is disabled by config. "
                        "Using %s.",
                        fallback_name,
                    )
                self._warned_fallback = True
            steering = self._compute_fallback(state, track_state, reference)

        max_delta = self.max_steer_rate * max(dt, 1.0e-6)
        steering = clamp(steering, self._previous_steer - max_delta, self._previous_steer + max_delta)
        steering = clamp(steering, -self.max_steer, self.max_steer)
        self._previous_steer = steering
        return VehicleCommand(
            steering_target=steering,
            gear_request=max(1, state.gear),
            command_timestamp=state.sim_time,
            command_valid_until=state.sim_time + dt,
        )

    def _compute_full_tube(
        self,
        state: VehicleState,
        track_state: TrackState,
        reference: ControllerReference,
        dt: float,
    ) -> float:
        assert self.tube_solver is not None
        width_left, width_right = self._track_widths_from_state(track_state)
        lateral_lower = -width_right + self.tube_boundary_margin
        lateral_upper = width_left - self.tube_boundary_margin
        heading_error = wrap_angle(state.yaw - reference.current_target_yaw)
        result = self.tube_solver.solve(
            lateral_error=track_state.n - reference.target_lateral_offset,
            heading_error=heading_error,
            steering_state=state.steering_angle,
            speed=state.speed,
            curvature=reference.current_target_curvature,
            curvature_preview=reference.curvature_preview,
            target_offset=reference.target_lateral_offset,
            lateral_lower=lateral_lower,
            lateral_upper=lateral_upper,
            previous_steer=self._previous_steer,
            dt=dt,
        )
        self.last_solver_status = result.status
        self.last_nominal_steering = result.nominal_steering
        self.last_ancillary_correction = result.ancillary_correction
        self.last_tube_state_bound = result.tube_state_bound.copy()
        self.last_tube_input_bound = result.tube_input_bound
        self.last_tube_radius = float(result.tube_state_bound[0])
        self.last_tightened_left = lateral_upper - self.last_tube_radius
        self.last_tightened_right = -lateral_lower - self.last_tube_radius
        return result.steering

    def _compute_fallback(
        self,
        state: VehicleState,
        track_state: TrackState,
        reference: ControllerReference,
    ) -> float:
        if self.fallback_method == "pure_pursuit":
            pure_pursuit = self._compute_pure_pursuit(state, reference)
            if pure_pursuit is not None:
                if self.tube_enabled:
                    self.last_tube_radius = self._tube_radius(state.speed)
                return pure_pursuit + self._fallback_boundary_correction(track_state)

        curvature = reference.target_curvature or track_state.curvature
        feedforward = math.atan(self.wheelbase * curvature)
        desired_n = self._safe_lateral_offset(state, track_state, reference.target_lateral_offset)
        lateral_error = track_state.n - desired_n
        boundary_correction = self._fallback_boundary_correction(track_state)
        return (
            feedforward
            - self.k_lat * lateral_error
            - self.k_head * track_state.heading_error
            + boundary_correction
        )

    def _compute_pure_pursuit(
        self,
        state: VehicleState,
        reference: ControllerReference,
    ) -> float | None:
        """Steer geometrically toward the preview point on the fixed path."""

        dx = reference.target_x - state.x
        dy = reference.target_y - state.y
        preview_distance = math.hypot(dx, dy)
        if preview_distance < max(self.fallback_min_preview_distance, 1.0e-6):
            return None
        target_bearing = math.atan2(dy, dx)
        alpha = wrap_angle(target_bearing - state.yaw)
        return math.atan2(
            2.0 * self.wheelbase * math.sin(alpha),
            preview_distance,
        )

    def _compute_osqp(
        self,
        state: VehicleState,
        track_state: TrackState,
        reference: ControllerReference,
        dt: float,
    ) -> float:
        import osqp  # type: ignore

        n = self.horizon_steps
        speed = max(state.speed, 0.5)
        dt_eff = dt if dt > 0.0 else self.dt
        curvature = reference.target_curvature or track_state.curvature
        target_lateral_offset = self._safe_lateral_offset(
            state,
            track_state,
            reference.target_lateral_offset,
        )
        a = np.array([[1.0, speed * dt_eff], [0.0, 1.0]], dtype=float)
        b = np.array([[0.0], [speed / self.wheelbase * dt_eff]], dtype=float)
        d = np.array([0.0, -speed * curvature * dt_eff], dtype=float)
        q = np.diag([self.q_lat, self.q_head])

        s_mat = np.zeros((2 * n, n), dtype=float)
        c_vec = np.zeros(2 * n, dtype=float)
        x = np.array(
            [track_state.n - target_lateral_offset, track_state.heading_error],
            dtype=float,
        )
        powers = [np.eye(2)]
        for _ in range(1, n + 1):
            powers.append(powers[-1] @ a)

        for k in range(n):
            x = a @ x + d
            c_vec[2 * k : 2 * k + 2] = x
            for j in range(k + 1):
                influence = powers[k - j] @ b
                s_mat[2 * k : 2 * k + 2, j] = influence[:, 0]

        q_bar = sparse.block_diag([q for _ in range(n)], format="csc")
        r = sparse.eye(n, format="csc") * self.r_steer
        rate = sparse.diags(
            diagonals=[np.ones(n), -np.ones(n - 1)],
            offsets=[0, -1],
            shape=(n, n),
            format="csc",
        )
        p = 2.0 * (
            sparse.csc_matrix(s_mat).T @ q_bar @ sparse.csc_matrix(s_mat)
            + r
            + self.r_rate * (rate.T @ rate)
        )
        q_vec = 2.0 * (sparse.csc_matrix(s_mat).T @ q_bar @ c_vec)

        identity = sparse.eye(n, format="csc")
        rate_limit = self.max_steer_rate * dt_eff
        lower = np.full(n, -self.max_steer)
        upper = np.full(n, self.max_steer)
        rate_lower = np.full(n, -rate_limit)
        rate_upper = np.full(n, rate_limit)
        rate_lower[0] += self._previous_steer
        rate_upper[0] += self._previous_steer
        constraint_blocks = [identity, rate]
        lower_blocks = [lower, rate_lower]
        upper_blocks = [upper, rate_upper]
        if self.tube_enabled:
            s_lat = sparse.csc_matrix(s_mat[0::2, :])
            c_lat = c_vec[0::2]
            n_lower, n_upper = self._tightened_lateral_bounds(track_state)
            constraint_blocks.append(s_lat)
            lower_blocks.append(np.full(n, n_lower - target_lateral_offset) - c_lat)
            upper_blocks.append(np.full(n, n_upper - target_lateral_offset) - c_lat)
        constraints = sparse.vstack(constraint_blocks, format="csc")
        l_bound = np.concatenate(lower_blocks)
        u_bound = np.concatenate(upper_blocks)

        solver = osqp.OSQP()
        solver.setup(P=p, q=np.asarray(q_vec).ravel(), A=constraints, l=l_bound, u=u_bound, verbose=False)
        result = solver.solve()
        status = str(getattr(result.info, "status", "")).lower()
        if result.x is None or "solved" not in status:
            LOGGER.warning(
                "OSQP failed to solve lateral MPC (%s). Falling back for this step.",
                getattr(result.info, "status", "unknown"),
            )
            return self._compute_fallback(state, track_state, reference)
        return float(result.x[0])

    def _safe_lateral_offset(
        self,
        state: VehicleState,
        track_state: TrackState,
        requested_offset: float,
    ) -> float:
        """Clamp planner offset into the robust tube corridor."""

        if not self.tube_enabled:
            return requested_offset
        self.last_tube_radius = self._tube_radius(state.speed)
        lower, upper = self._tightened_lateral_bounds(track_state)
        return clamp(requested_offset, lower, upper)

    def _tube_radius(self, speed: float) -> float:
        """Speed-dependent robust lateral error envelope in meters."""

        raw = self.tube_radius_base + self.tube_radius_speed_gain * max(0.0, speed)
        return clamp(raw, 0.0, self.tube_radius_max)

    def _track_widths_from_state(self, track_state: TrackState) -> tuple[float, float]:
        width_left = max(0.0, track_state.distance_left_boundary + track_state.n)
        width_right = max(0.0, track_state.distance_right_boundary - track_state.n)
        return width_left, width_right

    def _tightened_lateral_bounds(self, track_state: TrackState) -> tuple[float, float]:
        """Return absolute Frenet n limits after tube and safety margins."""

        width_left, width_right = self._track_widths_from_state(track_state)
        tightening = self.last_tube_radius + self.tube_boundary_margin
        lower = -width_right + tightening
        upper = width_left - tightening
        if lower > upper:
            center = 0.5 * (width_left - width_right)
            lower = center
            upper = center
        self.last_tightened_left = upper
        self.last_tightened_right = -lower
        return lower, upper

    def _fallback_boundary_correction(self, track_state: TrackState) -> float:
        """Add a steering nudge away from tightened boundaries for fallback mode."""

        if not self.tube_enabled:
            return 0.0
        lower, upper = self._tightened_lateral_bounds(track_state)
        correction = 0.0
        if track_state.n > upper:
            correction -= self.tube_soft_boundary_gain * (track_state.n - upper)
        if track_state.n < lower:
            correction += self.tube_soft_boundary_gain * (lower - track_state.n)
        return correction
