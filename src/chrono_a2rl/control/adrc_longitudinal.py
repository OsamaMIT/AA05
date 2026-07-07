"""Skeleton for future ADRC longitudinal control."""

from __future__ import annotations

from chrono_a2rl.common.types import ControllerReference, TrackState, VehicleCommand, VehicleState
from chrono_a2rl.control.controller_interface import Controller


class ADRCLongitudinalController(Controller):
    """Placeholder for active disturbance rejection speed control."""

    def reset(self) -> None:
        """Reset future observer state."""

    def compute_command(
        self,
        state: VehicleState,
        track_state: TrackState,
        reference: ControllerReference,
        dt: float,
    ) -> VehicleCommand:
        """Return a neutral command until ADRC is implemented."""

        del track_state, reference, dt
        # TODO: add ESO state estimation and disturbance-compensating control.
        return VehicleCommand(
            gear_request=max(1, state.gear),
            command_timestamp=state.sim_time,
            command_valid_until=state.sim_time,
        )
