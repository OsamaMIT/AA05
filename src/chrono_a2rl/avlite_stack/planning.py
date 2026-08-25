"""AVLite planning strategies for the TUMFTM Yas Marina race corridor."""

from __future__ import annotations

import math

from avlite.c10_perception.c11_perception_model import Map, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker

from chrono_a2rl.avlite_stack.settings import A2RLStackSettings
from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_loader import load_track_from_config


class A2RLGlobalRacePlanner(GlobalPlannerStrategy):
    """Build an AVLite GlobalPlan from the repository's raceline and limits."""

    def __init__(
        self,
        map: Map | None = None,
        setting: type[A2RLStackSettings] = A2RLStackSettings,
    ) -> None:
        super().__init__(map=map)
        self.setting = setting
        self.config = load_experiment_config(setting.experiment_config)
        self.track = load_track_from_config(self.config["track"])
        self.speed_profile = generate_speed_profile(
            self.track, self.config.get("speed_profile", {})
        )
        self.global_plan = self._build_plan()

    def plan(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> GlobalPlan:
        del perception_model, sensors
        self.global_plan = self._build_plan()
        return self.global_plan

    def _build_plan(self) -> GlobalPlan:
        path: list[tuple[float, float]] = []
        velocity: list[float] = []
        left_d: list[float] = []
        right_d: list[float] = []
        left_x: list[float] = []
        left_y: list[float] = []
        right_x: list[float] = []
        right_y: list[float] = []
        for index, s in enumerate(self.track.s_nodes):
            sample = self.track.interpolate(float(s))
            race_x, race_y = self.track.raceline_xy_at(float(s))
            race_n = self.track.raceline_lateral_offset_at(float(s))
            path.append((race_x, race_y))
            velocity.append(float(self.speed_profile.speed[index]))
            left_d.append(sample.width_left - race_n)
            right_d.append(-sample.width_right - race_n)
            normal_x = -math.sin(sample.heading)
            normal_y = math.cos(sample.heading)
            left_x.append(sample.x + sample.width_left * normal_x)
            left_y.append(sample.y + sample.width_left * normal_y)
            right_x.append(sample.x - sample.width_right * normal_x)
            right_y.append(sample.y - sample.width_right * normal_y)
        trajectory = TrajectoryTracker(path=path, velocity=velocity)
        return GlobalPlan(
            start_point=path[0],
            goal_point=path[-1],
            path=path,
            velocity=velocity,
            left_boundary_d=left_d,
            right_boundary_d=right_d,
            left_boundary_x=left_x,
            left_boundary_y=left_y,
            right_boundary_x=right_x,
            right_boundary_y=right_y,
            race_mode=True,
            trajectory=trajectory,
        )


class A2RLReferenceLocalPlanner(LocalPlanningStrategy):
    """Expose the global racing trajectory as AVLite's current LocalPlan."""

    def __init__(self, global_plan: GlobalPlan, env: PerceptionModel, setting=None):
        super().__init__(global_plan=global_plan, pm=env, setting=setting)
        self._local_plan = LocalPlan.from_trajectory(global_plan.trajectory)

    def replan(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> None:
        del sensors
        if perception_model is not None:
            self.pm = perception_model
        self._local_plan = LocalPlan.from_trajectory(self.global_trajectory)

    def get_local_plan(self) -> LocalPlan:
        return self._local_plan

    def set_global_plan(self, global_plan: GlobalPlan, ego_xy=None) -> None:
        super().set_global_plan(global_plan, ego_xy=ego_xy)
        self._local_plan = LocalPlan.from_trajectory(global_plan.trajectory)

    def reset(self, wp: int = 0) -> None:
        super().reset(wp)
        self._local_plan = LocalPlan.from_trajectory(self.global_trajectory)
