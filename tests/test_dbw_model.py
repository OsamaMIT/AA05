from __future__ import annotations

import numpy as np

from chrono_a2rl.chrono_interface.dbw_model import DBWModel
from chrono_a2rl.common.types import VehicleCommand, VehicleState


def test_dbw_clips_command_targets() -> None:
    dbw = DBWModel(
        {
            "max_steer": 0.4,
            "max_throttle": 1.0,
            "max_brake": 1.0,
            "max_steer_rate": 100.0,
            "steering_time_constant": 0.0,
            "throttle_time_constant": 0.0,
            "brake_time_constant": 0.0,
        }
    )
    applied = dbw.apply(
        VehicleCommand(steering_target=2.0, throttle_target=3.0, brake_target=0.0),
        VehicleState(),
        0.1,
    )
    assert applied.steering_target == 0.4
    assert applied.throttle_target == 1.0
    assert dbw.last_saturated


def test_dbw_limits_steering_rate() -> None:
    dbw = DBWModel(
        {
            "max_steer": 1.0,
            "max_steer_rate": 1.0,
            "steering_time_constant": 0.0,
            "throttle_time_constant": 0.0,
            "brake_time_constant": 0.0,
        }
    )
    applied = dbw.apply(VehicleCommand(steering_target=1.0), VehicleState(), 0.1)
    assert np.isclose(applied.steering_target, 0.1)
    assert dbw.last_saturated


def test_dbw_prevents_throttle_brake_conflict() -> None:
    dbw = DBWModel(
        {
            "max_steer_rate": 100.0,
            "steering_time_constant": 0.0,
            "throttle_time_constant": 0.0,
            "brake_time_constant": 0.0,
        }
    )
    applied = dbw.apply(
        VehicleCommand(throttle_target=0.4, brake_target=0.7),
        VehicleState(),
        0.1,
    )
    assert applied.throttle_target == 0.0
    assert applied.brake_target == 0.7


def test_dbw_brake_command_clears_lingering_throttle() -> None:
    dbw = DBWModel(
        {
            "max_steer_rate": 100.0,
            "steering_time_constant": 0.0,
            "throttle_time_constant": 0.2,
            "brake_time_constant": 0.2,
            "brake_priority": True,
        }
    )
    state = VehicleState(throttle=0.8, brake=0.0)

    applied = dbw.apply(
        VehicleCommand(throttle_target=0.0, brake_target=0.5),
        state,
        0.02,
    )

    assert applied.throttle_target == 0.0
    assert applied.brake_target > 0.0
