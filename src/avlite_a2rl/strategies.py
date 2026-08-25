"""Plugin-owned subclasses that register the A2RL strategies with AVLite."""

from chrono_a2rl.avlite_stack.controller import (
    A2RLMPCControlStrategy as _A2RLMPCControlStrategy,
)
from chrono_a2rl.avlite_stack.planning import (
    A2RLGlobalRacePlanner as _A2RLGlobalRacePlanner,
    A2RLReferenceLocalPlanner as _A2RLReferenceLocalPlanner,
)
from chrono_a2rl.avlite_stack.tasks import (
    A2RLLapTelemetryTask as _A2RLLapTelemetryTask,
)
from chrono_a2rl.avlite_stack.world_bridge import (
    A2RLChronoWorldBridge as _A2RLChronoWorldBridge,
)


class A2RLChronoWorldBridge(_A2RLChronoWorldBridge):
    """AVLite-registered Chrono world bridge."""


class A2RLGlobalRacePlanner(_A2RLGlobalRacePlanner):
    """AVLite-registered TUMFTM global race planner."""


class A2RLReferenceLocalPlanner(_A2RLReferenceLocalPlanner):
    """AVLite-registered fixed-raceline local planner."""


class A2RLMPCControlStrategy(_A2RLMPCControlStrategy):
    """AVLite-registered lateral MPC and speed PID controller."""


class A2RLLapTelemetryTask(_A2RLLapTelemetryTask):
    """AVLite-registered lap timing and telemetry task."""


__all__ = [
    "A2RLChronoWorldBridge",
    "A2RLGlobalRacePlanner",
    "A2RLLapTelemetryTask",
    "A2RLMPCControlStrategy",
    "A2RLReferenceLocalPlanner",
]
