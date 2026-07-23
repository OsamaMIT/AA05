from __future__ import annotations

from chrono_a2rl.common.config import load_experiment_config


def test_experiment_overrides_are_available_at_top_level() -> None:
    config = load_experiment_config("configs/experiments/rl_planner_yas_marina.yaml")

    assert config["speed_profile"]["max_speed"] == 83.3333333333
    assert config["speed_profile"]["max_lateral_accel"] == 19.0
    assert config["reward"]["completion_bonus"] == 1000.0
    assert config["experiment"]["speed_profile"]["max_speed"] == 83.3333333333
    assert config["rl"]["action_mode"] == "profile_pedal_residual"
    assert config["rl"]["longitudinal_action_deadband"] == 0.05
    assert config["rl"]["speed_demand_curvature_source"] == "raceline"
    assert config["speed_profile"]["curvature_source"] == "raceline"
    assert config["rl"]["model_dir"] == "models/ppo_profile_speed_residual"
    assert config["rl"]["profile_speed_residual_authority"] == 0.08
    assert config["rl"]["longitudinal_action_rise_rate"] == 6.0
    assert config["rl"]["longitudinal_action_fall_rate"] == 10.0
    assert config["rl"]["use_sde"] is True
    assert config["rl"]["sde_sample_freq"] == 25
    assert config["rl"]["gamma"] == 0.9995
    assert config["rl"]["gae_lambda"] == 0.995
    assert config["termination"]["start_finish_s"] == 0.0
    assert config["termination"]["minimum_lap_fraction"] == 0.95
    assert config["reward"]["trail_braking_lookahead_m"] == 350.0
    assert config["reward"]["trail_braking_max_reference"] == 0.70
    assert config["reward"]["trail_braking_taper_exponent"] == 0.80
