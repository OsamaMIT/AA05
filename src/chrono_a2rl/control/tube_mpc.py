"""Robust Tube MPC solver for lateral path tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import solve_discrete_are


@dataclass(slots=True)
class TubeMPCResult:
    """One robust Tube MPC solution and its diagnostics."""

    steering: float
    nominal_steering: float
    ancillary_correction: float
    nominal_state: np.ndarray
    actual_error: np.ndarray
    tube_state_bound: np.ndarray
    tube_input_bound: float
    status: str
    objective: float


class TubeMPCSolver:
    """Nominal MPC plus ancillary LQR and an iterated RPI disturbance tube."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        wheelbase: float,
        max_steer: float,
        max_steer_rate: float,
    ) -> None:
        self.horizon = max(2, int(config.get("horizon_steps", 20)))
        self.default_dt = float(config.get("dt", 0.02))
        self.wheelbase = float(wheelbase)
        self.max_steer = float(max_steer)
        self.max_steer_rate = float(max_steer_rate)
        self.state_size = 3
        self.q = np.diag(
            [
                float(config.get("weight_lateral_error", 12.0)),
                float(config.get("weight_heading_error", 3.5)),
                float(config.get("weight_steering_state", 0.15)),
            ]
        )
        self.r = max(float(config.get("weight_steering", 0.05)), 1.0e-8)
        self.r_rate = max(float(config.get("weight_steering_rate", 0.8)), 0.0)
        self.feedback_q = np.diag(
            [
                float(config.get("tube_feedback_weight_lateral", 0.5)),
                float(config.get("tube_feedback_weight_heading", 1.0)),
                float(config.get("tube_feedback_weight_steering", 0.1)),
            ]
        )
        self.feedback_r = max(
            float(config.get("tube_feedback_weight_input", 2.0)),
            1.0e-8,
        )
        self.max_ancillary_correction = max(
            float(config.get("tube_max_ancillary_correction", self.max_steer)),
            0.0,
        )
        self.heading_limit = float(config.get("tube_heading_limit", 0.55))
        self.disturbance_bound = np.asarray(
            [
                float(config.get("tube_disturbance_lateral", 0.002)),
                float(config.get("tube_disturbance_heading", 0.0005)),
                float(config.get("tube_disturbance_steering", 0.0002)),
            ],
            dtype=float,
        )
        self.steering_time_constant = max(
            float(config.get("tube_steering_time_constant", 0.06)),
            0.0,
        )
        self.rpi_iterations = max(1, int(config.get("tube_rpi_iterations", 80)))
        self.rpi_tolerance = max(float(config.get("tube_rpi_tolerance", 1.0e-8)), 0.0)
        self.minimum_nominal_rate_fraction = float(
            config.get("tube_minimum_nominal_rate_fraction", 0.15)
        )
        self.tighten_steering_rate = bool(
            config.get("tube_tighten_steering_rate", False)
        )
        self.verbose = bool(config.get("osqp_verbose", False))
        self.polishing = bool(config.get("osqp_polishing", False))
        self._nominal_state: np.ndarray | None = None
        self._previous_target_offset: float | None = None
        self._previous_nominal_steer = 0.0
        self.last_result: TubeMPCResult | None = None

    def reset(self) -> None:
        """Reset the carried nominal trajectory."""

        self._nominal_state = None
        self._previous_target_offset = None
        self._previous_nominal_steer = 0.0
        self.last_result = None

    def solve(
        self,
        *,
        lateral_error: float,
        heading_error: float,
        steering_state: float,
        speed: float,
        curvature: float,
        curvature_preview: tuple[float, ...] = (),
        target_offset: float,
        lateral_lower: float,
        lateral_upper: float,
        previous_steer: float,
        dt: float,
    ) -> TubeMPCResult:
        """Solve the robust nominal QP and apply ancillary feedback."""

        import osqp  # type: ignore

        dt_eff = float(dt if dt > 0.0 else self.default_dt)
        speed_eff = max(float(speed), 0.5)
        a, b, disturbance = self.system_matrices(
            speed=speed_eff,
            curvature=curvature,
            dt=dt_eff,
        )
        curvature_sequence = np.full(self.horizon, float(curvature), dtype=float)
        preview_count = min(len(curvature_preview), self.horizon)
        if preview_count:
            curvature_sequence[:preview_count] = np.asarray(
                curvature_preview[:preview_count],
                dtype=float,
            )
            curvature_sequence[preview_count:] = curvature_sequence[preview_count - 1]
        disturbance_sequence = np.zeros((self.horizon, self.state_size), dtype=float)
        disturbance_sequence[:, 1] = -speed_eff * curvature_sequence * dt_eff
        feedback_gain = self.lqr_gain(a, b)
        closed_loop = a - b @ feedback_gain
        tube_state_bound, tube_input_bound, tube_rate_bound = self.rpi_bounds(
            closed_loop,
            feedback_gain,
        )

        target_lower = lateral_lower + tube_state_bound[0]
        target_upper = lateral_upper - tube_state_bound[0]
        if target_lower > target_upper:
            raise RuntimeError("Tube MPC lateral corridor is empty after robust tightening")
        absolute_lateral_position = lateral_error + target_offset
        tightened_target = float(np.clip(target_offset, target_lower, target_upper))
        target_offset = tightened_target
        actual_error = np.asarray(
            [
                absolute_lateral_position - target_offset,
                heading_error,
                steering_state,
            ],
            dtype=float,
        )
        nominal_state = self._initial_nominal_state(
            actual_error,
            target_offset=target_offset,
            tube_state_bound=tube_state_bound,
        )
        tube_error = actual_error - nominal_state
        raw_ancillary = float(-(feedback_gain @ tube_error)[0])
        ancillary_limit = min(tube_input_bound, self.max_ancillary_correction)
        ancillary = float(
            np.clip(raw_ancillary, -ancillary_limit, ancillary_limit)
        )

        s_matrix, c_vector = self._condensed_prediction(
            a,
            b,
            disturbance_sequence,
            nominal_state,
        )
        q_bar = sparse.block_diag([self.q for _ in range(self.horizon)], format="csc")
        s_sparse = sparse.csc_matrix(s_matrix)
        difference = self._difference_matrix()
        previous_nominal_vector = np.zeros(self.horizon, dtype=float)
        previous_nominal_vector[0] = self._previous_nominal_steer
        p_matrix = 2.0 * (
            s_sparse.T @ q_bar @ s_sparse
            + self.r * sparse.eye(self.horizon, format="csc")
            + self.r_rate * (difference.T @ difference)
        )
        q_vector = 2.0 * (
            np.asarray(s_sparse.T @ q_bar @ c_vector).reshape(-1)
            - self.r_rate
            * np.asarray(difference.T @ previous_nominal_vector).reshape(-1)
        )

        constraints: list[sparse.csc_matrix] = []
        lower_bounds: list[np.ndarray] = []
        upper_bounds: list[np.ndarray] = []

        nominal_steer_limit = self.max_steer - tube_input_bound
        if nominal_steer_limit <= 0.0:
            raise RuntimeError("Tube MPC steering corridor is empty after robust tightening")
        constraints.append(sparse.eye(self.horizon, format="csc"))
        lower_bounds.append(np.full(self.horizon, -nominal_steer_limit))
        upper_bounds.append(np.full(self.horizon, nominal_steer_limit))

        first_selector = sparse.csc_matrix(
            ([1.0], ([0], [0])),
            shape=(1, self.horizon),
        )
        rate_limit = self.max_steer_rate * dt_eff
        constraints.append(first_selector)
        lower_bounds.append(np.asarray([previous_steer - rate_limit - ancillary]))
        upper_bounds.append(np.asarray([previous_steer + rate_limit - ancillary]))

        if self.horizon > 1:
            future_difference = self._future_difference_matrix()
            robust_rate_limit = (
                max(
                    rate_limit - tube_rate_bound,
                    rate_limit * self.minimum_nominal_rate_fraction,
                )
                if self.tighten_steering_rate
                else rate_limit
            )
            constraints.append(future_difference)
            lower_bounds.append(np.full(self.horizon - 1, -robust_rate_limit))
            upper_bounds.append(np.full(self.horizon - 1, robust_rate_limit))

        lateral_prediction = sparse.csc_matrix(
            s_matrix[0 :: self.state_size, :]
        )
        lateral_free = c_vector[0 :: self.state_size]
        robust_lower_error = target_lower - target_offset
        robust_upper_error = target_upper - target_offset
        constraints.append(lateral_prediction)
        lower_bounds.append(
            np.full(self.horizon, robust_lower_error) - lateral_free
        )
        upper_bounds.append(
            np.full(self.horizon, robust_upper_error) - lateral_free
        )

        heading_prediction = sparse.csc_matrix(
            s_matrix[1 :: self.state_size, :]
        )
        heading_free = c_vector[1 :: self.state_size]
        robust_heading_limit = max(
            self.heading_limit - tube_state_bound[1],
            0.05,
        )
        constraints.append(heading_prediction)
        lower_bounds.append(
            np.full(self.horizon, -robust_heading_limit) - heading_free
        )
        upper_bounds.append(
            np.full(self.horizon, robust_heading_limit) - heading_free
        )

        steering_prediction = sparse.csc_matrix(
            s_matrix[2 :: self.state_size, :]
        )
        steering_free = c_vector[2 :: self.state_size]
        robust_steering_state_limit = max(
            self.max_steer - tube_state_bound[2],
            0.01,
        )
        constraints.append(steering_prediction)
        lower_bounds.append(
            np.full(self.horizon, -robust_steering_state_limit) - steering_free
        )
        upper_bounds.append(
            np.full(self.horizon, robust_steering_state_limit) - steering_free
        )

        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(p_matrix),
            q=q_vector,
            A=sparse.vstack(constraints, format="csc"),
            l=np.concatenate(lower_bounds),
            u=np.concatenate(upper_bounds),
            verbose=self.verbose,
            polishing=self.polishing,
            warm_starting=True,
        )
        solution = solver.solve(raise_error=False)
        status = str(getattr(solution.info, "status", "unknown")).lower()
        if solution.x is None or "solved" not in status:
            raise RuntimeError(f"Tube MPC QP failed: {status}")

        nominal_sequence = np.asarray(solution.x, dtype=float)
        nominal_steering = float(nominal_sequence[0])
        steering = nominal_steering + ancillary
        predicted = c_vector + s_matrix @ nominal_sequence
        self._nominal_state = predicted[: self.state_size].copy()
        self._previous_target_offset = float(target_offset)
        self._previous_nominal_steer = nominal_steering
        result = TubeMPCResult(
            steering=steering,
            nominal_steering=nominal_steering,
            ancillary_correction=ancillary,
            nominal_state=nominal_state.copy(),
            actual_error=actual_error,
            tube_state_bound=tube_state_bound,
            tube_input_bound=tube_input_bound,
            status=status,
            objective=float(getattr(solution.info, "obj_val", 0.0)),
        )
        self.last_result = result
        return result

    def system_matrices(
        self,
        *,
        speed: float,
        curvature: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the discrete linear lateral-error model."""

        actuator_alpha = (
            1.0
            if self.steering_time_constant <= 0.0
            else dt / (self.steering_time_constant + dt)
        )
        a = np.asarray(
            [
                [1.0, speed * dt, 0.0],
                [0.0, 1.0, speed / self.wheelbase * dt],
                [0.0, 0.0, 1.0 - actuator_alpha],
            ],
            dtype=float,
        )
        b = np.asarray(
            [[0.0], [0.0], [actuator_alpha]],
            dtype=float,
        )
        disturbance = np.asarray(
            [0.0, -speed * curvature * dt, 0.0],
            dtype=float,
        )
        return a, b, disturbance

    def lqr_gain(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the stabilizing discrete ancillary-feedback gain."""

        feedback_r = np.asarray([[self.feedback_r]], dtype=float)
        p = solve_discrete_are(a, b, self.feedback_q, feedback_r)
        return np.linalg.solve(
            feedback_r + b.T @ p @ b,
            b.T @ p @ a,
        )

    def rpi_bounds(
        self,
        closed_loop: np.ndarray,
        feedback_gain: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Bound the iterated robust invariant zonotope in state and input."""

        generator = np.diag(self.disturbance_bound)
        state_bound = np.zeros(self.state_size, dtype=float)
        input_bound = 0.0
        rate_bound = 0.0
        identity = np.eye(self.state_size)
        for _ in range(self.rpi_iterations):
            state_increment = np.sum(np.abs(generator), axis=1)
            input_increment = float(np.sum(np.abs(feedback_gain @ generator)))
            rate_increment = float(
                np.sum(np.abs(feedback_gain @ (closed_loop - identity) @ generator))
            )
            state_bound += state_increment
            input_bound += input_increment
            rate_bound += rate_increment
            if max(
                float(np.max(state_increment)),
                input_increment,
            ) <= self.rpi_tolerance:
                break
            generator = closed_loop @ generator
        rate_bound += float(
            np.sum(np.abs(feedback_gain) * self.disturbance_bound[None, :])
        )
        return state_bound, input_bound, rate_bound

    def _initial_nominal_state(
        self,
        actual_error: np.ndarray,
        *,
        target_offset: float,
        tube_state_bound: np.ndarray,
    ) -> np.ndarray:
        if self._nominal_state is None:
            return actual_error.copy()
        nominal = self._nominal_state.copy()
        if self._previous_target_offset is not None:
            nominal[0] += self._previous_target_offset - target_offset
        error = actual_error - nominal
        contained_error = np.clip(
            error,
            -tube_state_bound,
            tube_state_bound,
        )
        return actual_error - contained_error

    def _condensed_prediction(
        self,
        a: np.ndarray,
        b: np.ndarray,
        disturbance: np.ndarray,
        initial_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        s_matrix = np.zeros(
            (self.state_size * self.horizon, self.horizon),
            dtype=float,
        )
        c_vector = np.zeros(self.state_size * self.horizon, dtype=float)
        state_free = initial_state.copy()
        input_influence = np.zeros(
            (self.state_size, self.horizon),
            dtype=float,
        )
        disturbance_sequence = (
            np.repeat(disturbance[None, :], self.horizon, axis=0)
            if disturbance.ndim == 1
            else disturbance
        )
        for step in range(self.horizon):
            state_free = a @ state_free + disturbance_sequence[step]
            input_influence = a @ input_influence
            input_influence[:, step] += b[:, 0]
            start = self.state_size * step
            stop = start + self.state_size
            c_vector[start:stop] = state_free
            s_matrix[start:stop, :] = input_influence
        return s_matrix, c_vector

    def _difference_matrix(self) -> sparse.csc_matrix:
        rows = np.arange(self.horizon)
        matrix = np.eye(self.horizon)
        matrix[rows[1:], rows[:-1]] = -1.0
        return sparse.csc_matrix(matrix)

    def _future_difference_matrix(self) -> sparse.csc_matrix:
        matrix = np.zeros((self.horizon - 1, self.horizon), dtype=float)
        rows = np.arange(self.horizon - 1)
        matrix[rows, rows] = -1.0
        matrix[rows, rows + 1] = 1.0
        return sparse.csc_matrix(matrix)
