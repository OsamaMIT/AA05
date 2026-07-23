from __future__ import annotations

import argparse
import os
from pathlib import Path

from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.rl.train import (
    _add_training_override_args,
    _apply_training_overrides,
    _ppo_resume_kwargs,
    _training_overrides_from_args,
    resolve_resume_checkpoint,
)


def test_training_cli_overrides_update_config() -> None:
    parser = argparse.ArgumentParser()
    _add_training_override_args(parser)
    args = parser.parse_args(
        [
            "--total-timesteps",
            "1234",
            "--n-envs",
            "3",
            "--vec-env-type",
            "dummy",
            "--backend",
            "mock",
            "--seed",
            "99",
            "--no-randomize-resets",
            "--reset-speed-mode",
            "profile",
            "--reset-speed-scale-min",
            "0.8",
            "--reset-speed-scale-max",
            "1.1",
            "--longitudinal-action-deadband",
            "0.08",
            "--longitudinal-action-rise-rate",
            "5.0",
            "--longitudinal-action-fall-rate",
            "9.0",
            "--profile-speed-residual-authority",
            "0.06",
            "--profile-speed-residual-error-guard-mps",
            "4.0",
            "--vehicle-max-accel",
            "13.5",
            "--max-lateral-accel",
            "20.0",
            "--max-episode-time",
            "45.0",
        ]
    )
    overrides = _training_overrides_from_args(args)
    config = load_experiment_config("configs/experiments/rl_planner_yas_marina.yaml")
    _apply_training_overrides(config, overrides)

    assert config["rl"]["total_timesteps"] == 1234
    assert config["rl"]["n_envs"] == 3
    assert config["rl"]["vec_env_type"] == "dummy"
    assert config["rl"]["train_backend"] == "mock"
    assert config["rl"]["seed"] == 99
    assert config["rl"]["randomize_resets"] is False
    assert config["rl"]["reset_speed_mode"] == "profile"
    assert config["rl"]["reset_speed_scale_min"] == 0.8
    assert config["rl"]["reset_speed_scale_max"] == 1.1
    assert config["rl"]["longitudinal_action_deadband"] == 0.08
    assert config["rl"]["longitudinal_action_rise_rate"] == 5.0
    assert config["rl"]["longitudinal_action_fall_rate"] == 9.0
    assert config["rl"]["profile_speed_residual_authority"] == 0.06
    assert config["rl"]["profile_speed_residual_error_guard_mps"] == 4.0
    assert config["vehicle"]["max_accel"] == 13.5
    assert config["speed_profile"]["max_lateral_accel"] == 20.0
    assert config["simulation"]["max_episode_time"] == 45.0


def test_training_cli_resume_latest_override() -> None:
    parser = argparse.ArgumentParser()
    _add_training_override_args(parser)
    args = parser.parse_args(["--resume", "latest"])
    overrides = _training_overrides_from_args(args)

    assert overrides["resume_from"] == "latest"

    alias_args = parser.parse_args(["--resume-latest"])
    assert _training_overrides_from_args(alias_args)["resume_from"] == "latest"


def test_resolve_resume_checkpoint_uses_newest_matching_policy(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    old_checkpoint = model_dir / "ppo_planner_policy_100_steps.zip"
    new_checkpoint = model_dir / "ppo_planner_policy_200_steps.zip"
    other_policy_checkpoint = model_dir / "ppo_speed_policy_999_steps.zip"
    final_model = model_dir / "final_model.zip"
    for path in (old_checkpoint, new_checkpoint, other_policy_checkpoint, final_model):
        path.write_text("placeholder")

    os.utime(old_checkpoint, ns=(100, 100))
    os.utime(new_checkpoint, ns=(200, 200))
    os.utime(other_policy_checkpoint, ns=(400, 400))
    os.utime(final_model, ns=(300, 300))

    assert (
        resolve_resume_checkpoint("latest", model_dir, "ppo_planner_policy")
        == final_model
    )


def test_ppo_resume_kwargs_include_overridden_optimizer_settings() -> None:
    kwargs = _ppo_resume_kwargs(
        {
            "n_steps": 1024,
            "batch_size": 256,
            "learning_rate": 0.0002,
            "ent_coef": 0.005,
            "clip_range": 0.15,
            "target_kl": 0.02,
            "use_sde": True,
            "sde_sample_freq": 25,
        }
    )

    assert kwargs["n_steps"] == 1024
    assert kwargs["batch_size"] == 256
    assert kwargs["learning_rate"] == 0.0002
    assert kwargs["ent_coef"] == 0.005
    assert kwargs["clip_range"] == 0.15
    assert kwargs["target_kl"] == 0.02
    assert kwargs["use_sde"] is True
    assert kwargs["sde_sample_freq"] == 25
