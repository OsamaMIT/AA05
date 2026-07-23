# Architecture

```text
configs + vehicle params
    -> track loader and generated/optimized speed profile
    -> ChronoDirectBackend
    -> DBW model and safety supervisor
    -> fast raceline steering + RL throttle/brake
    -> lap runner and metrics
    -> Gymnasium speed-policy wrapper
```

## Direct Chrono Training

Training calls the simulator directly because policy optimization needs
deterministic reset/step control, high throughput, and simple failure handling.
ROS is excellent for runtime integration, but it adds event timing and transport
concerns that are unnecessary in the critical training loop.

## Isolation Boundaries

- `chrono_interface/` owns all Chrono-specific imports and backend details.
- `track/` owns geometry, Frenet projection, curbs, and speed profiles.
- `control/` owns controller interfaces, MPC/PID implementations, and safety.
- `rl/` owns Gymnasium observations, rewards, environments, and SB3 training.
- `evaluation/` owns metrics, replay, repeatable-profile optimization, and
  future disturbance testing.

## RL Longitudinal Mode

The planner environment fixes the geometric Yas Marina TUMFTM raceline and lets
RL output one signed longitudinal action. Positive values command throttle,
negative values command brake, and values inside the deadband coast. Lateral
The default training profile uses tube-aware pure pursuit for steering while
DBW applies actuator lag and physical output limits. The separate full Tube MPC
profile uses a constrained OSQP nominal controller and LQR ancillary feedback
for later fine-tuning and evaluation.

The curvature-derived or optimized raceline speed profile is tracked by the
speed PID. PPO receives the profile as preview information and supplies only a
small guarded pedal residual, so it can fine-tune acceleration and braking
without replacing baseline longitudinal control or steering.

## Current Backend

`ChronoDirectBackend` always exposes the stable API:

```python
reset(initial_state=None)
step(command, dt)
get_state()
close()
```

When `backend: chrono`, `ChronoDirectBackend` first tries
`PyChronoKinematicBackend`. That backend creates a PyChrono system, fixed flat
ground, and chassis body, then mirrors the scaffold kinematic vehicle state into
Chrono each step. If PyChrono cannot initialize, it falls back to
`MockChronoBackend`.

The next fidelity step is replacing the kinematic Chrono body internals with a
full `pychrono.vehicle` wheeled vehicle while preserving the same backend API.

## Vehicle Profile

The vehicle configuration is public-reference A2RL EAV24/EAV25 style. It
records the public Dallara autonomous Super Formula platform, K20C1-based
engine, 3MO gearbox, Brembo electro-hydraulic carbon brakes, Yokohama Advan
tires, and sensor/compute suite. Exact dynamics values are explicitly marked as
simulation estimates.
