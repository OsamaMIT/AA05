"""A2RL-style rigid chassis representation inside a PyChrono system."""

from __future__ import annotations

import math

from chrono_a2rl.common.types import VehicleState


class A2RLChronoBody:
    """Own a simplified formula-car rigid body and synchronize planar state."""

    def __init__(self, chrono, system, body) -> None:
        self.chrono = chrono
        self.system = system
        self.body = body

    def sync(self, state: VehicleState) -> None:
        self.body.SetPos(self._vec(state.x, state.y, state.z))
        self.body.SetRot(self._yaw_quaternion(state.yaw))
        if hasattr(self.body, "SetPosDt"):
            self.body.SetPosDt(self._vec(state.vx, state.vy, state.vz))
        if hasattr(self.body, "SetAngVelParent"):
            self.body.SetAngVelParent(self._vec(0.0, 0.0, state.yaw_rate))

    def _vec(self, x: float, y: float, z: float):
        cls = getattr(self.chrono, "ChVector3d", None) or getattr(
            self.chrono, "ChVectorD", None
        )
        return cls(float(x), float(y), float(z))

    def _yaw_quaternion(self, yaw: float):
        if hasattr(self.chrono, "QuatFromAngleZ"):
            return self.chrono.QuatFromAngleZ(float(yaw))
        cls = getattr(self.chrono, "ChQuaterniond", None) or getattr(
            self.chrono, "ChQuaternionD", None
        )
        return cls(math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))

