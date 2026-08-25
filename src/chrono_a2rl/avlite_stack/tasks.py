"""AVLite execution tasks for lap timing and physics telemetry logging."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from avlite.c40_execution.c43_task_strategy import TaskSchedule, TaskStrategy
import pandas as pd
import yaml

from chrono_a2rl.avlite_stack.settings import A2RLStackSettings, repository_path
from chrono_a2rl.avlite_stack.world_bridge import A2RLChronoWorldBridge
from chrono_a2rl.evaluation.metrics import (
    compute_metrics,
    format_lap_time,
    metrics_to_dict,
)


class A2RLLapTelemetryTask(TaskStrategy):
    """Record AVLite stack state and stop after a validated closed-loop lap."""

    schedule = TaskSchedule.EVERY_CYCLE

    def __init__(self, setting: type[A2RLStackSettings] = A2RLStackSettings) -> None:
        self.setting = setting
        self.rows: list[dict[str, Any]] = []
        self.laps = 0
        self.progress_m = 0.0
        self.previous_s: float | None = None
        self.offtrack_streak = 0
        self.last_outputs: dict[str, str] = {}

    def execute(self, executer, event=None) -> None:
        del event
        world = executer.world
        if not isinstance(world, A2RLChronoWorldBridge):
            return
        state = world.get_vehicle_state()
        track_state = world.last_track_state
        telemetry = world.get_vehicle_telemetry()
        command = world.last_vehicle_command
        if self.previous_s is None:
            self.previous_s = track_state.s
        ds = track_state.s - self.previous_s
        if ds < -0.5 * world.track.length:
            ds += world.track.length
        elif ds > 0.5 * world.track.length:
            ds -= world.track.length
        self.progress_m += max(ds, 0.0)
        crossed = world.track.crossed_line_forward(
            self.previous_s, track_state.s, line_s=0.0
        )
        self.previous_s = track_state.s
        if crossed and self.progress_m >= 0.95 * world.track.length:
            self.laps += 1

        self.offtrack_streak = (
            self.offtrack_streak + 1 if not track_state.on_track else 0
        )
        row = {
            "step": len(self.rows),
            "sim_time": format_lap_time(state.sim_time),
            "sim_time_seconds": state.sim_time,
            "episode_time_seconds": state.sim_time,
            "x": state.x,
            "y": state.y,
            "yaw": state.yaw,
            "speed_kmh": state.speed * 3.6,
            "yaw_rate": state.yaw_rate,
            "steering_angle": state.steering_angle,
            "throttle": state.throttle,
            "brake": state.brake,
            "s": track_state.s,
            "progress_s": self.progress_m,
            "lateral_error": track_state.n,
            "heading_error": track_state.heading_error,
            "on_track": track_state.on_track,
            "on_curb": track_state.on_curb,
            "curb_penalty_weight": track_state.curb_penalty_weight,
            "target_speed_kmh": world.speed_profile.speed_at(track_state.s) * 3.6,
            "steering_target": command.steering_target,
            "throttle_target": command.throttle_target,
            "brake_target": command.brake_target,
            "control_saturated": world.last_control_saturated,
            "termination_reason": "",
        }
        row.update(telemetry.to_dict())
        self.rows.append(row)

        reason = ""
        if self.laps >= max(int(self.setting.stop_after_laps), 1):
            reason = "lap_completed"
        elif (
            bool(self.setting.stop_on_offtrack)
            and self.offtrack_streak >= int(self.setting.max_offtrack_steps)
        ):
            reason = "off_track"
        if reason:
            self.rows[-1]["termination_reason"] = reason
            self.last_outputs = self._save(reason)
            executer.stop()

    def reset(self) -> None:
        self.rows = []
        self.laps = 0
        self.progress_m = 0.0
        self.previous_s = None
        self.offtrack_streak = 0
        self.last_outputs = {}

    def finish(self, reason: str) -> dict[str, str]:
        """Persist a partial rollout when the runtime stops outside this task."""

        if self.last_outputs:
            return self.last_outputs
        if self.rows:
            self.rows[-1]["termination_reason"] = reason
        self.last_outputs = self._save(reason)
        return self.last_outputs

    def _save(self, reason: str) -> dict[str, str]:
        output_dir = repository_path(self.setting.log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"avlite_a2rl_{stamp}.csv"
        metrics_path = output_dir / f"avlite_a2rl_{stamp}_metrics.yaml"
        pd.DataFrame(self.rows).to_csv(csv_path, index=False)
        metrics = metrics_to_dict(compute_metrics(self.rows, reason))
        metrics["avlite_profile"] = "a2rl_chrono_yas_marina"
        metrics_path.write_text(
            yaml.safe_dump(metrics, sort_keys=False), encoding="utf-8"
        )
        return {"log_csv": str(csv_path), "metrics_yaml": str(metrics_path)}
