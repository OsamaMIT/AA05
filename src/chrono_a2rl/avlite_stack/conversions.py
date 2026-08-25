"""Conversions between AVLite stack datatypes and racing-domain dataclasses."""

from __future__ import annotations

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame

from chrono_a2rl.common.types import VehicleState


def ego_to_vehicle_state(
    ego: EgoState,
    *,
    sensors: SensorFrame | None = None,
    steering_angle: float = 0.0,
    throttle: float = 0.0,
    brake: float = 0.0,
    gear: int = 1,
    sim_time: float = 0.0,
) -> VehicleState:
    yaw_rate = 0.0
    if sensors is not None and sensors.wheel_odometry is not None:
        yaw_rate = float(sensors.wheel_odometry.yaw_rate)
    return VehicleState(
        x=float(ego.x),
        y=float(ego.y),
        z=float(ego.z),
        yaw=float(ego.theta),
        vx=float(ego.velocity),
        speed=max(float(ego.velocity), 0.0),
        yaw_rate=yaw_rate,
        steering_angle=float(steering_angle),
        throttle=float(throttle),
        brake=float(brake),
        gear=int(gear),
        sim_time=float(sim_time),
    )


def update_ego_from_vehicle(ego: EgoState, state: VehicleState) -> None:
    """Mutate AVLite's authoritative ego object without replacing its identity."""

    ego.x = float(state.x)
    ego.y = float(state.y)
    ego.z = float(state.z)
    ego.theta = float(state.yaw)
    ego.velocity = float(state.speed)

