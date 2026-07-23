from __future__ import annotations

import pytest

from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.rl.callbacks import stable_baselines3_available
from chrono_a2rl.rl.train import _make_training_env


@pytest.mark.skipif(not stable_baselines3_available(), reason="Stable-Baselines3 is optional")
def test_make_training_env_uses_vector_env() -> None:
    config_path = "configs/experiments/rl_planner_yas_marina.yaml"
    config = load_experiment_config(config_path)
    config["simulation"]["backend"] = "mock"
    config["rl"]["n_envs"] = 2
    config["rl"]["vec_env_type"] = "dummy"
    config["rl"]["randomize_resets"] = True

    env = _make_training_env(config)
    try:
        assert env.num_envs == 2
        obs = env.reset()
        assert obs.shape[0] == 2
    finally:
        env.close()
