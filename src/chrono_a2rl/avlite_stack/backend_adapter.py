"""Legacy backend-shaped adapter over the canonical AVLite world bridge."""

from __future__ import annotations

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel

from chrono_a2rl.avlite_stack.world_bridge import A2RLChronoWorldBridge
from chrono_a2rl.common.types import VehicleCommand, VehicleState
from chrono_a2rl.vehicle.telemetry import VehicleTelemetry


class AVLiteBackendAdapter:
    """Let existing Gym/evaluation code step through an AVLite WorldBridge."""

    def __init__(self, config: dict) -> None:
        ego = EgoState()
        self.world = A2RLChronoWorldBridge(
            ego_state=ego,
            pm=PerceptionModel(ego_vehicle=ego),
            resolved_config=config,
        )

    def reset(self, initial_state: VehicleState | None = None) -> VehicleState:
        return self.world.reset_vehicle(initial_state)

    def step(self, command: VehicleCommand, dt: float) -> VehicleState:
        return self.world.apply_vehicle_command(command, dt)

    def get_state(self) -> VehicleState:
        return self.world.get_vehicle_state()

    def get_telemetry(self) -> VehicleTelemetry:
        return self.world.get_vehicle_telemetry()

    @property
    def last_control_saturated(self) -> bool:
        return self.world.last_control_saturated

    def close(self) -> None:
        self.world.close()

