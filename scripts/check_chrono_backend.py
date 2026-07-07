#!/usr/bin/env python3
"""Check whether the PyChrono backend can initialize and step."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chrono_a2rl.chrono_interface.direct_backend import ChronoDirectBackend
from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.common.types import VehicleCommand, VehicleState


def main() -> None:
    config = load_experiment_config("configs/experiments/mpc_yas_marina_flat.yaml")
    config["simulation"]["backend"] = "chrono"
    backend = ChronoDirectBackend(config["vehicle"], config["simulation"])
    state = backend.reset(VehicleState(speed=1.0))
    state = backend.step(VehicleCommand(throttle_target=0.2), config["simulation"]["control_dt"])
    backend.close()
    print(f"backend={backend._backend.__class__.__name__}")  # noqa: SLF001
    print(f"sim_time={state.sim_time:.3f}")
    print(f"speed={state.speed:.3f}")


if __name__ == "__main__":
    main()
