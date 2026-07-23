"""Common interface for swappable vehicle dynamics implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chrono_a2rl.common.types import VehicleCommand, VehicleState
from chrono_a2rl.vehicle.telemetry import VehicleTelemetry


class VehicleDynamicsModel(ABC):
    @abstractmethod
    def reset(self, initial_state: VehicleState | None = None) -> VehicleState:
        """Reset model state."""

    @abstractmethod
    def step(self, command: VehicleCommand, dt: float) -> VehicleState:
        """Advance model state."""

    @abstractmethod
    def get_state(self) -> VehicleState:
        """Return a detached state snapshot."""

    @abstractmethod
    def get_telemetry(self) -> VehicleTelemetry:
        """Return current physics telemetry."""

