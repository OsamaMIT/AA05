from __future__ import annotations

import numpy as np
import pytest

from chrono_a2rl.common.types import ControllerReference, TrackState, VehicleState
from chrono_a2rl.control.mpc_lateral import LateralMPCController
from chrono_a2rl.control.tube_mpc import TubeMPCSolver


def test_tube_mpc_clamps_reference_inside_tightened_corridor() -> None:
    controller = LateralMPCController(
        {
            "mode": "tube",
            "tube_enabled": True,
            "use_osqp": False,
            "tube_radius_base": 0.5,
            "tube_radius_speed_gain": 0.0,
            "tube_boundary_margin": 0.5,
        },
        {"wheelbase": 3.0, "max_steer": 0.5, "max_steer_rate": 10.0},
    )
    state = VehicleState(speed=20.0)
    track_state = TrackState(
        n=4.8,
        distance_left_boundary=0.2,
        distance_right_boundary=10.8,
    )

    safe_offset = controller._safe_lateral_offset(state, track_state, requested_offset=5.5)

    assert np.isclose(safe_offset, 4.0)
    assert np.isclose(controller.last_tightened_left, 4.0)
    assert np.isclose(controller.last_tightened_right, 5.0)


def test_tube_fallback_steers_away_from_left_boundary() -> None:
    controller = LateralMPCController(
        {
            "mode": "tube",
            "tube_enabled": True,
            "use_osqp": False,
            "tube_radius_base": 0.5,
            "tube_radius_speed_gain": 0.0,
            "tube_boundary_margin": 0.5,
            "tube_soft_boundary_gain": 0.3,
        },
        {"wheelbase": 3.0, "max_steer": 0.5, "max_steer_rate": 10.0},
    )
    command = controller.compute_command(
        VehicleState(speed=20.0),
        TrackState(
            n=4.8,
            distance_left_boundary=0.2,
            distance_right_boundary=10.8,
        ),
        ControllerReference(target_lateral_offset=5.5),
        dt=0.02,
    )

    assert command.steering_target < 0.0


def test_pure_pursuit_fallback_steers_toward_preview_point() -> None:
    controller = LateralMPCController(
        {
            "mode": "tube",
            "tube_enabled": True,
            "use_osqp": False,
            "fallback_method": "pure_pursuit",
        },
        {"wheelbase": 3.0, "max_steer": 0.5, "max_steer_rate": 10.0},
    )
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, speed=40.0)
    track_state = TrackState(
        distance_left_boundary=6.0,
        distance_right_boundary=6.0,
    )

    left = controller.compute_command(
        state,
        track_state,
        ControllerReference(target_x=20.0, target_y=3.0),
        dt=0.02,
    )
    controller.reset()
    right = controller.compute_command(
        state,
        track_state,
        ControllerReference(target_x=20.0, target_y=-3.0),
        dt=0.02,
    )

    assert left.steering_target > 0.0
    assert right.steering_target < 0.0


def test_explicit_pure_pursuit_mode_does_not_request_osqp() -> None:
    controller = LateralMPCController(
        {
            "mode": "pure_pursuit",
            "use_osqp": False,
            "fallback_method": "pure_pursuit",
        },
        {"wheelbase": 3.0, "max_steer": 0.5, "max_steer_rate": 10.0},
    )

    command = controller.compute_command(
        VehicleState(x=0.0, y=0.0, yaw=0.0, speed=20.0),
        TrackState(distance_left_boundary=6.0, distance_right_boundary=6.0),
        ControllerReference(target_x=20.0, target_y=2.0),
        dt=0.02,
    )

    assert command.steering_target > 0.0
    assert controller.last_solver_status == "pure_pursuit"
    assert controller._warned_fallback is False


def _tube_solver() -> TubeMPCSolver:
    return TubeMPCSolver(
        {
            "horizon_steps": 15,
            "dt": 0.02,
            "weight_lateral_error": 12.0,
            "weight_heading_error": 3.5,
            "weight_steering": 0.05,
            "weight_steering_rate": 0.8,
            "tube_disturbance_lateral": 0.002,
            "tube_disturbance_heading": 0.0005,
            "tube_rpi_iterations": 80,
        },
        wheelbase=3.0,
        max_steer=0.38,
        max_steer_rate=1.5,
    )


def test_ancillary_lqr_stabilizes_lateral_error_model() -> None:
    solver = _tube_solver()
    a, b, _ = solver.system_matrices(speed=30.0, curvature=0.01, dt=0.02)
    gain = solver.lqr_gain(a, b)
    eigenvalues = np.linalg.eigvals(a - b @ gain)

    assert np.max(np.abs(eigenvalues)) < 1.0
    assert gain.shape == (1, 3)


def test_iterated_rpi_tube_produces_finite_constraint_tightening() -> None:
    solver = _tube_solver()
    a, b, _ = solver.system_matrices(speed=30.0, curvature=0.01, dt=0.02)
    gain = solver.lqr_gain(a, b)

    state_bound, input_bound, rate_bound = solver.rpi_bounds(a - b @ gain, gain)

    assert np.all(state_bound > 0.0)
    assert np.all(np.isfinite(state_bound))
    assert 0.0 < input_bound < solver.max_steer
    assert rate_bound > 0.0


def test_full_tube_mpc_solves_nominal_qp_and_applies_ancillary_feedback() -> None:
    pytest.importorskip("osqp")
    solver = _tube_solver()
    first = solver.solve(
        lateral_error=0.1,
        heading_error=0.02,
        steering_state=0.0,
        speed=30.0,
        curvature=0.005,
        target_offset=0.0,
        lateral_lower=-6.0,
        lateral_upper=6.0,
        previous_steer=0.0,
        dt=0.02,
    )
    assert "solved" in first.status
    assert abs(first.steering) <= solver.max_steer
    assert abs(first.steering) <= solver.max_steer_rate * 0.02 + 1.0e-5
    assert solver._nominal_state is not None

    nominal_next = solver._nominal_state.copy()
    second = solver.solve(
        lateral_error=float(nominal_next[0] + 0.001),
        heading_error=float(nominal_next[1]),
        steering_state=float(nominal_next[2]),
        speed=30.0,
        curvature=0.005,
        target_offset=0.0,
        lateral_lower=-6.0,
        lateral_upper=6.0,
        previous_steer=first.steering,
        dt=0.02,
    )

    assert "solved" in second.status
    assert second.ancillary_correction < 0.0
    assert abs(second.ancillary_correction) <= second.tube_input_bound
    assert second.tube_input_bound > 0.0
