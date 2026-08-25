"""Select the canonical simulation boundary for an experiment."""

from __future__ import annotations

from typing import Any

from chrono_a2rl.chrono_interface.direct_backend import ChronoDirectBackend


def create_simulation_backend(config: dict[str, Any]):
    """Create an AVLite bridge adapter by default, or the legacy direct facade."""

    avlite_config = config.get("avlite", {})
    if bool(avlite_config.get("enabled", True)):
        from chrono_a2rl.avlite_stack.backend_adapter import AVLiteBackendAdapter

        return AVLiteBackendAdapter(config)
    return ChronoDirectBackend(config.get("vehicle", {}), config.get("simulation", {}))
