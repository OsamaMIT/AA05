"""Validate the exported plugin in a fresh process outside the source tree."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_community_plugin", ROOT / "scripts/build_community_plugin.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_export_refuses_to_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        builder.build_plugin(tmp_path)


def test_standalone_discovery_settings_and_mpc(tmp_path):
    plugin = builder.build_plugin(tmp_path / builder.NAME)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["AVLITE_PLUGINS_DIR"] = str(tmp_path)
    env["AVLITE_CONFIG_DIR"] = str(tmp_path / "user_config")
    code = r'''
import math
import sys
from avlite.c60_apps.c63_plugins import sync_community_plugins, load_plugin_settings_class
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
sync_community_plugins({"avlite_a2rl_mpc": "avlite_a2rl_mpc"})
from avlite.plugins import avlite_a2rl_mpc as plugin
settings = load_plugin_settings_class("avlite_a2rl_mpc", sys.argv[1])
assert settings is plugin.PluginSettings
assert ControlStrategy.registry["A2RLTubeMPCControlStrategy"] is plugin.A2RLTubeMPCControlStrategy
planner = plugin.A2RLMPCGlobalRacePlanner()
plan = planner.plan()
assert len(plan.path) == 1110
assert max(plan.velocity) <= 15.0
local = plugin.A2RLMPCReferenceLocalPlanner(plan, PerceptionModel())
local.replan()
assert local.get_local_plan() is not None
controller = plugin.A2RLTubeMPCControlStrategy()
assert controller.lateral.full_tube_enabled
assert controller.lateral.tube_solver is not None
x, y = plan.path[0]
nx, ny = plan.path[1]
ego = EgoState(x=x, y=y, theta=math.atan2(ny-y, nx-x), velocity=12.0)
for _ in range(5):
    command = controller.control(ego, plan, control_dt=0.02)
    assert math.isfinite(command.steer)
    assert abs(command.steer) <= controller.lateral.max_steer
    assert 0 <= command.throttle <= 1
    assert 0 <= command.brake <= 1
assert "solved" in controller.lateral.last_solver_status.lower(), controller.lateral.last_solver_status
controller.reset()
assert controller.lateral.last_solver_status == "not_started"
assert not any(m == "chrono_a2rl" or m.startswith("chrono_a2rl.") for m in sys.modules)
print("Standalone plugin: discovery, settings, Yas Marina, OSQP and reset passed")
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(plugin)], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
