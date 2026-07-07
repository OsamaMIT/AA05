"""State extraction hooks for future real Chrono integration."""

from __future__ import annotations

from chrono_a2rl.common.types import VehicleState


def extract_mock_state(state: VehicleState) -> VehicleState:
    """Return a state object from the mock backend path."""

    return state
