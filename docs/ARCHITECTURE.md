# Architecture

```text
configs + vehicle params
    -> track loader and speed profile
    -> ChronoDirectBackend
    -> DBW model and safety supervisor
    -> lateral MPC + speed PID
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
- `evaluation/` owns metrics, replay, and future disturbance testing.

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
