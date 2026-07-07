# RL Design

## First Policy

The first Stable-Baselines3 policy is deliberately high level. Its action is:

```text
target_speed_scale
```

The environment multiplies the curvature-based speed profile by this scale.
MPC still controls steering, and PID still controls throttle/brake.

## Observation

The initial observation vector is:

```text
speed
target_speed
lateral_error
heading_error
yaw_rate
steering_angle
curvature
distance_left_boundary
distance_right_boundary
previous_action
progress_s
```

## Reward

The first reward is intentionally simple:

```text
progress_reward
- offtrack_penalty
- instability_penalty
- excessive_control_penalty
```

This makes early training behavior interpretable and keeps unsafe low-level
control actions out of the policy space.

## SB3 Integration

`src/chrono_a2rl/rl/train.py` uses PPO and saves checkpoints under `models/`.
Stable-Baselines3 is optional; missing dependencies produce a clear install
message.
