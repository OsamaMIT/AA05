"""Repository-owned settings used by the AVLite A2RL plugin."""

from __future__ import annotations

from pathlib import Path

from chrono_a2rl.common.config import REPO_ROOT


class A2RLStackSettings:
    """Mutable settings singleton loaded from the AVLite plugin profile section."""

    filepath: str = "configs/plugin_avlite_a2rl.yaml"
    experiment_config: str = "configs/experiments/rl_planner_yas_marina.yaml"
    backend: str = "configured"
    chrono_mode: str = "configured"
    start_s_m: float = 0.0
    initial_speed_mps: float = 12.0
    log_dir: str = "logs/avlite"
    stop_after_laps: int = 1
    stop_on_offtrack: bool = True
    max_offtrack_steps: int = 20
    control_command_timeout_s: float = 0.10


def repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
