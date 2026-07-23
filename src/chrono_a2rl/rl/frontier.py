"""Shared progress-frontier state and environment role helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


START_LINE_ROLE = "start_line"
FRONTIER_PRACTICE_ROLE = "frontier_practice"
RANDOM_ROLE = "random"
EVALUATION_ROLE = "evaluation"


@dataclass(slots=True)
class ProgressFrontierState:
    """Serializable curriculum state shared by parallel environments."""

    schema_version: int = 1
    run_id: str = ""
    frontier_progress_m: float = 350.0
    best_validated_progress_m: float = 0.0
    update_count: int = 0
    total_timesteps: int = 0

    def advance(
        self,
        candidate_progress_m: float,
        *,
        maximum_advance_m: float = 150.0,
        track_length: float | None = None,
    ) -> bool:
        """Advance monotonically toward a validated start-line result."""

        candidate = max(0.0, float(candidate_progress_m))
        self.best_validated_progress_m = max(self.best_validated_progress_m, candidate)
        upper = self.frontier_progress_m + max(0.0, maximum_advance_m)
        next_frontier = min(candidate, upper)
        if track_length is not None:
            next_frontier = min(next_frontier, float(track_length))
        if next_frontier <= self.frontier_progress_m + 1.0e-9:
            return False
        self.frontier_progress_m = next_frontier
        self.update_count += 1
        return True


def assign_training_role(
    env_index: int,
    n_envs: int,
    config: dict[str, Any] | None = None,
) -> str:
    """Assign deterministic start/frontier/random roles to vector environments."""

    cfg = config or {}
    start_count = int(cfg.get("frontier_start_envs", 2))
    practice_count = int(cfg.get("frontier_practice_envs", 4))
    random_count = int(cfg.get("frontier_random_envs", 2))
    configured_total = max(1, start_count + practice_count + random_count)
    if n_envs != configured_total:
        start_count = max(1, round(n_envs * start_count / configured_total))
        random_count = max(0, round(n_envs * random_count / configured_total))
        if start_count + random_count > n_envs:
            random_count = max(0, n_envs - start_count)
        practice_count = max(0, n_envs - start_count - random_count)

    if env_index < start_count:
        return START_LINE_ROLE
    if env_index < start_count + practice_count:
        return FRONTIER_PRACTICE_ROLE
    return RANDOM_ROLE


def save_frontier_state(path: str | Path, state: ProgressFrontierState) -> Path:
    """Atomically write curriculum state beside a model checkpoint."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(asdict(state), sort_keys=False), encoding="utf-8")
    temporary.replace(output)
    return output


def load_frontier_state(path: str | Path) -> ProgressFrontierState:
    """Load curriculum state and reject unsupported schemas."""

    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    state = ProgressFrontierState(**data)
    if state.schema_version != 1:
        raise ValueError(f"Unsupported frontier state schema: {state.schema_version}")
    return state


def frontier_sidecar_path(model_path: str | Path) -> Path:
    """Return the frontier-state path paired with an SB3 model archive."""

    path = Path(model_path)
    stem = path.stem if path.suffix == ".zip" else path.name
    return path.with_name(f"{stem}_frontier.yaml")
