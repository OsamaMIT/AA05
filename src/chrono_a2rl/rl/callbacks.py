"""Stable-Baselines3 callback helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chrono_a2rl.rl.frontier import (
    FRONTIER_PRACTICE_ROLE,
    RANDOM_ROLE,
    START_LINE_ROLE,
    ProgressFrontierState,
    frontier_sidecar_path,
    save_frontier_state,
)

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover - exercised when optional RL extras are absent
    BaseCallback = object  # type: ignore[misc,assignment]


def stable_baselines3_available() -> bool:
    """Return whether SB3 can be imported."""

    try:
        import stable_baselines3  # noqa: F401
    except ImportError:
        return False
    return True


class ProgressFrontierCallback(BaseCallback):  # type: ignore[misc]
    """Synchronize frontier state and save model/state checkpoint pairs."""

    def __init__(
        self,
        *,
        state: ProgressFrontierState,
        run_dir: str | Path,
        checkpoint_prefix: str,
        checkpoint_interval_steps: int,
        maximum_advance_m: float,
        track_length: float,
        verbose: int = 0,
    ) -> None:
        if not stable_baselines3_available():
            raise RuntimeError("Stable-Baselines3 is required for ProgressFrontierCallback")
        super().__init__(verbose=verbose)
        self.state = state
        self.run_dir = Path(run_dir)
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_interval_steps = max(1, int(checkpoint_interval_steps))
        self.maximum_advance_m = float(maximum_advance_m)
        self.track_length = float(track_length)
        self.last_checkpoint_step = int(state.total_timesteps)

    def _on_training_start(self) -> None:
        self.training_env.env_method(
            "set_progress_frontier",
            self.state.frontier_progress_m,
        )
        self._record_frontier()

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = list(self.locals.get("infos", []))
        dones = self.locals.get("dones", [])
        self.state.total_timesteps = int(self.num_timesteps)
        candidates = [
            float(info.get("validated_progress_m", 0.0))
            for info, done in zip(infos, dones)
            if bool(done) and info.get("training_role") == START_LINE_ROLE
        ]
        advanced = False
        previous_frontier = self.state.frontier_progress_m
        if candidates:
            advanced = self.state.advance(
                max(candidates),
                maximum_advance_m=self.maximum_advance_m,
                track_length=self.track_length,
            )
            if advanced:
                self.training_env.env_method(
                    "set_progress_frontier",
                    self.state.frontier_progress_m,
                )
                save_frontier_state(self.run_dir / "frontier_state.yaml", self.state)

        for info in infos:
            self.logger.record_mean(
                "longitudinal/pedal_action",
                float(info.get("longitudinal_action", 0.0)),
            )
            self.logger.record_mean(
                "longitudinal/effective_pedal_action",
                float(info.get("effective_longitudinal_action", 0.0)),
            )
            self.logger.record_mean(
                "longitudinal/throttle",
                float(info.get("applied_throttle", 0.0)),
            )
            self.logger.record_mean(
                "longitudinal/brake",
                float(info.get("applied_brake", 0.0)),
            )
            self.logger.record_mean(
                "longitudinal/action_change",
                float(info.get("longitudinal_action_change", 0.0)),
            )
            self.logger.record_mean(
                "profile/speed_error_kmh",
                float(info.get("profile_speed_error_kmh", 0.0)),
            )
            self.logger.record_mean(
                "profile/absolute_speed_error_kmh",
                abs(float(info.get("profile_speed_error_kmh", 0.0))),
            )
            self.logger.record_mean(
                "profile/pid_throttle",
                float(info.get("profile_pid_throttle", 0.0)),
            )
            self.logger.record_mean(
                "profile/pid_brake",
                float(info.get("profile_pid_brake", 0.0)),
            )
            self.logger.record_mean(
                "profile/residual_pedal",
                float(info.get("profile_residual_pedal", 0.0)),
            )
            self.logger.record_mean(
                "profile/residual_guard",
                float(info.get("profile_residual_guard", 0.0)),
            )
            self.logger.record_mean(
                "corner/braking_demand",
                float(info.get("speed_corner_strength", 0.0)),
            )
            self.logger.record_mean(
                "corner/overspeed_fraction",
                float(info.get("corner_overspeed_fraction", 0.0)),
            )
            self.logger.record_mean(
                "corner/actual_deceleration",
                float(info.get("actual_deceleration", 0.0)),
            )
            self.logger.record_mean(
                "corner/controlled_braking_reward",
                float(info.get("corner_controlled_braking_reward", 0.0)),
            )
            self.logger.record_mean(
                "corner/overspeed_penalty",
                float(info.get("corner_overspeed_penalty", 0.0)),
            )
            self.logger.record_mean(
                "corner/excessive_braking_penalty",
                float(info.get("corner_excessive_braking_penalty", 0.0)),
            )
            self.logger.record_mean(
                "corner/braking_reward_applied",
                float(info.get("corner_braking_reward_applied", 0.0)),
            )
            self.logger.record_mean(
                "trail/active",
                float(bool(info.get("trail_braking_active", False))),
            )
            self.logger.record_mean(
                "trail/target_brake",
                float(info.get("trail_brake_target", 0.0)),
            )
            self.logger.record_mean(
                "trail/applied_brake",
                float(info.get("trail_brake_applied", 0.0)),
            )
            self.logger.record_mean(
                "trail/alignment_error",
                float(info.get("trail_brake_alignment_error", 0.0)),
            )
            self.logger.record_mean(
                "trail/alignment_reward",
                float(info.get("trail_brake_alignment_reward", 0.0)),
            )
            self.logger.record_mean(
                "trail/missing_brake_penalty",
                float(info.get("trail_brake_missing_penalty", 0.0)),
            )
            self.logger.record_mean(
                "trail/excess_reference_penalty",
                float(info.get("trail_brake_excess_reference_penalty", 0.0)),
            )
            self.logger.record_mean(
                "trail/release_quality",
                float(info.get("trail_brake_release_quality", 1.0)),
            )
            self.logger.record_mean(
                "frontier/validated_progress_m",
                float(info.get("validated_progress_m", 0.0)),
            )
            self.logger.record_mean(
                "corner/completion_rate",
                float(bool(info.get("corner_completed", False))),
            )
            self.logger.record_mean(
                "corner/heading_completion",
                float(info.get("corner_heading_completion", 0.0)),
            )
            role = str(info.get("training_role", ""))
            self.logger.record_mean("role/start_line", float(role == START_LINE_ROLE))
            self.logger.record_mean(
                "role/frontier_practice",
                float(role == FRONTIER_PRACTICE_ROLE),
            )
            self.logger.record_mean("role/random", float(role == RANDOM_ROLE))
            if bool(info.get("corner_completed", False)):
                self.logger.record_mean(
                    "corner/score",
                    float(info.get("corner_score", 0.0)),
                )
                self.logger.record_mean(
                    "corner/apex_speed_kmh",
                    float(info.get("apex_speed_kmh", 0.0)),
                )
                self.logger.record_mean(
                    "corner/exit_speed_kmh",
                    float(info.get("exit_speed_kmh", 0.0)),
                )
            if float(info.get("kinetic_crash_penalty", 0.0)) > 0.0:
                self.logger.record_mean(
                    "crash/kinetic_penalty",
                    float(info["kinetic_crash_penalty"]),
                )
        if self.num_timesteps - self.last_checkpoint_step >= self.checkpoint_interval_steps:
            self.save_checkpoint(self.num_timesteps)
            self.last_checkpoint_step = int(self.num_timesteps)
        self._record_frontier()
        if candidates:
            self.logger.record("frontier/max_start_progress_m", max(candidates))
        self.logger.record("frontier/advanced", float(advanced))
        self.logger.record(
            "frontier/advancement_m",
            self.state.frontier_progress_m - previous_frontier,
        )
        return True

    def _on_training_end(self) -> None:
        self.state.total_timesteps = int(self.num_timesteps)
        save_frontier_state(self.run_dir / "frontier_state.yaml", self.state)

    def save_checkpoint(self, timestep: int) -> Path:
        """Save an SB3 archive and the exact frontier state that accompanies it."""

        model_path = self.run_dir / f"{self.checkpoint_prefix}_{int(timestep)}_steps.zip"
        self.model.save(str(model_path))
        self.state.total_timesteps = int(timestep)
        save_frontier_state(frontier_sidecar_path(model_path), self.state)
        return model_path

    def save_final_state(self, model_path: str | Path) -> Path:
        """Pair final_model.zip with current frontier state."""

        self.state.total_timesteps = int(self.num_timesteps)
        save_frontier_state(self.run_dir / "frontier_state.yaml", self.state)
        return save_frontier_state(frontier_sidecar_path(model_path), self.state)

    def _record_frontier(self) -> None:
        self.logger.record("frontier/progress_m", self.state.frontier_progress_m)
        self.logger.record(
            "frontier/best_validated_progress_m",
            self.state.best_validated_progress_m,
        )
        self.logger.record("frontier/update_count", self.state.update_count)
