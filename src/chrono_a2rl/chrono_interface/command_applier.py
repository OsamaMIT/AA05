"""Command application hooks for future real Chrono integration."""

from __future__ import annotations

from chrono_a2rl.chrono_interface.dbw_model import DBWModel
from chrono_a2rl.common.types import VehicleCommand, VehicleState


def apply_dbw_command(
    dbw: DBWModel,
    command: VehicleCommand,
    state: VehicleState,
    dt: float,
) -> VehicleCommand:
    """Apply DBW limits to a command."""

    return dbw.apply(command, state, dt)
