"""Controller interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chrono_a2rl.common.types import (
    ControllerReference,
    VehicleCommand,
    VehicleState,
    TrackState,
)


class Controller(ABC):
    """Generic controller interface for low-level DBW commands."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state."""

    @abstractmethod
    def compute_command(
        self,
        state: VehicleState,
        track_state: TrackState,
        reference: ControllerReference,
        dt: float,
    ) -> VehicleCommand:
        """Compute a DBW-style vehicle command."""
