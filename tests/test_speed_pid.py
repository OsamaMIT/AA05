from __future__ import annotations

from chrono_a2rl.common.types import ControllerReference, TrackState, VehicleState
from chrono_a2rl.control.speed_pid import SpeedPIDController


def _controller() -> SpeedPIDController:
    return SpeedPIDController(
        {
            "kp_throttle": 1.0,
            "ki_throttle": 0.03,
            "kd_throttle": 0.0,
            "kp_brake": 1.0,
            "ki_brake": 0.02,
            "kd_brake": 0.0,
            "coast_enabled": True,
            "coast_underspeed_band_mps": 0.25,
            "coast_overspeed_band_mps": 1.5,
            "coast_integral_decay_rate": 4.0,
        }
    )


def test_small_overspeed_uses_coast_instead_of_brake() -> None:
    controller = _controller()

    command = controller.compute_command(
        VehicleState(speed=30.0),
        TrackState(),
        ControllerReference(target_speed=29.0),
        0.02,
    )

    assert command.throttle_target == 0.0
    assert command.brake_target == 0.0
    assert controller.last_mode == "coast"


def test_large_profile_drop_still_requests_brake() -> None:
    controller = _controller()

    command = controller.compute_command(
        VehicleState(speed=30.0),
        TrackState(),
        ControllerReference(target_speed=25.0),
        0.02,
    )

    assert command.throttle_target == 0.0
    assert command.brake_target > 0.0
    assert controller.last_mode == "brake"


def test_coast_band_does_not_block_acceleration() -> None:
    controller = _controller()

    command = controller.compute_command(
        VehicleState(speed=30.0),
        TrackState(),
        ControllerReference(target_speed=32.0),
        0.02,
    )

    assert command.throttle_target > 0.0
    assert command.brake_target == 0.0
    assert controller.last_mode == "throttle"
