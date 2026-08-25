"""AVLite control payloads used by the A2RL bridge."""

from __future__ import annotations

from dataclasses import dataclass

from avlite.c30_control.c31_control_model import AckermannControlCommand


@dataclass
class A2RLAckermannCommand(AckermannControlCommand):
    """Ackermann command with explicit drive-by-wire pedal targets."""

    throttle: float = 0.0
    brake: float = 0.0
    gear: int = 1
    emergency_brake: bool = False

