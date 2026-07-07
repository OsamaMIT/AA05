"""First lateral MPC controller with proportional fallback."""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np
from scipy import sparse

from chrono_a2rl.common.logging import get_logger
from chrono_a2rl.common.math_utils import clamp
from chrono_a2rl.common.types import ControllerReference, TrackState, VehicleCommand, VehicleState
from chrono_a2rl.control.controller_interface import Controller


LOGGER = get_logger(__name__)


class LateralMPCController(Controller):
    """Linearized bicycle lateral MPC.

    If OSQP is unavailable, the controller uses a deterministic proportional
    steering law with feedforward curvature.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        vehicle_config: dict[str, Any] | None = None,
    ) -> None:
        cfg = config or {}
        vehicle = vehicle_config or {}
        self.horizon_steps = int(cfg.get("horizon_steps", 15))
        self.dt = float(cfg.get("dt", 0.02))
        self.q_lat = float(cfg.get("weight_lateral_error", 8.0))
        self.q_head = float(cfg.get("weight_heading_error", 2.5))
        self.r_steer = float(cfg.get("weight_steering", 0.05))
        self.r_rate = float(cfg.get("weight_steering_rate", 0.6))
        self.k_lat = float(cfg.get("fallback_lateral_gain", 0.18))
        self.k_head = float(cfg.get("fallback_heading_gain", 1.15))
        self.wheelbase = float(vehicle.get("wheelbase", 2.97))
        self.max_steer = float(vehicle.get("max_steer", 0.45))
        self.max_steer_rate = float(vehicle.get("max_steer_rate", 1.2))
        self.osqp_requested = bool(cfg.get("use_osqp", True))
        self.osqp_available = importlib.util.find_spec("osqp") is not None
        self.use_osqp = self.osqp_requested and self.osqp_available
        self._previous_steer = 0.0
        self._warned_fallback = False

    def reset(self) -> None:
        self._previous_steer = 0.0

    def compute_command(
        self,
        state: VehicleState,
        track_state: TrackState,
        reference: ControllerReference,
        dt: float,
    ) -> VehicleCommand:
        if self.use_osqp:
            steering = self._compute_osqp(state, track_state, reference, dt)
        else:
            if not self._warned_fallback:
                if self.osqp_requested and not self.osqp_available:
                    LOGGER.warning("OSQP is not installed. Using proportional lateral MPC fallback.")
                else:
                    LOGGER.warning(
                        "OSQP lateral MPC is disabled by config. "
                        "Using proportional lateral MPC fallback."
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

    def _compute_fallback(
        self,
        state: VehicleState,
        track_state: TrackState,
        reference: ControllerReference,
    ) -> float:
        curvature = reference.target_curvature or track_state.curvature
        feedforward = math.atan(self.wheelbase * curvature)
        return feedforward - self.k_lat * track_state.n - self.k_head * track_state.heading_error

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
        a = np.array([[1.0, speed * dt_eff], [0.0, 1.0]], dtype=float)
        b = np.array([[0.0], [speed / self.wheelbase * dt_eff]], dtype=float)
        d = np.array([0.0, -speed * curvature * dt_eff], dtype=float)
        q = np.diag([self.q_lat, self.q_head])

        s_mat = np.zeros((2 * n, n), dtype=float)
        c_vec = np.zeros(2 * n, dtype=float)
        x = np.array([track_state.n, track_state.heading_error], dtype=float)
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
        constraints = sparse.vstack([identity, rate], format="csc")
        l_bound = np.concatenate([lower, rate_lower])
        u_bound = np.concatenate([upper, rate_upper])

        solver = osqp.OSQP()
        solver.setup(P=p, q=np.asarray(q_vec).ravel(), A=constraints, l=l_bound, u=u_bound, verbose=False)
        result = solver.solve()
        if result.x is None:
            LOGGER.warning("OSQP failed to solve lateral MPC. Falling back for this step.")
            return self._compute_fallback(state, track_state, reference)
        return float(result.x[0])
