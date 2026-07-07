"""Stable-Baselines3 training entrypoint for the first speed policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.rl.callbacks import stable_baselines3_available
from chrono_a2rl.rl.envs.chrono_racing_env import ChronoRacingEnv


def train(config_path: str | Path) -> None:
    """Train PPO speed policy if Stable-Baselines3 is installed."""

    if not stable_baselines3_available():
        raise RuntimeError(
            "Stable-Baselines3 is not installed. Install optional RL dependencies with "
            "`python3 -m pip install -e .[rl]` or `python3 -m pip install stable-baselines3`."
        )

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    config = load_experiment_config(config_path)
    rl_cfg = config.get("rl", {})
    env = ChronoRacingEnv(config=config)
    model_dir = Path(str(rl_cfg.get("model_dir", "models/ppo_speed_policy")))
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=int(rl_cfg.get("checkpoint_interval_steps", 10000)),
        save_path=str(model_dir),
        name_prefix="ppo_speed_policy",
    )
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=int(rl_cfg.get("n_steps", 1024)),
        batch_size=int(rl_cfg.get("batch_size", 256)),
        learning_rate=float(rl_cfg.get("learning_rate", 3.0e-4)),
        gamma=float(rl_cfg.get("gamma", 0.99)),
        gae_lambda=float(rl_cfg.get("gae_lambda", 0.95)),
        clip_range=float(rl_cfg.get("clip_range", 0.2)),
        tensorboard_log=str(rl_cfg.get("tensorboard_log", "logs/tensorboard")),
        verbose=1,
    )
    model.learn(
        total_timesteps=int(rl_cfg.get("total_timesteps", 100000)),
        callback=checkpoint_callback,
    )
    model.save(str(model_dir / "final_model"))
    env.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/rl_speed_policy_yas_marina.yaml",
        help="RL experiment config path.",
    )
    args = parser.parse_args(argv)
    train(args.config)


if __name__ == "__main__":
    main(sys.argv[1:])
