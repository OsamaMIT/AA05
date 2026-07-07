# Chrono A2RL Training

Research scaffold for an A2RL-style autonomous racing training stack:

```text
Project Chrono direct backend
    -> A2RL-style vehicle model
    -> flat Yas Marina track from TUMFTM racetrack-database
    -> DBW-style command interface
    -> MPC controller baseline
    -> Gymnasium environment
    -> Stable-Baselines3 high-level RL policy
    -> ROS 2 runtime integration later
```

The repository is useful before Project Chrono is installed. If Chrono data or
bindings are unavailable, `ChronoDirectBackend` uses a mock kinematic bicycle
simulator. Yas Marina is processed from `TUMFTM/racetrack-database`; the
synthetic oval is now only a fallback if those processed files are removed.

The vehicle profile is based on public A2RL EAV24/EAV25 information: Dallara
autonomous Super Formula platform, K20C1-based turbo engine, 3MO 6-speed
gearbox, Brembo electro-hydraulic carbon brakes, Yokohama Advan tires, and the
published camera/radar/lidar/compute stack. Exact dynamics values remain
documented simulation estimates.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .[dev]
```

Optional extras:

```bash
python3 -m pip install -e .[mpc]   # OSQP
python3 -m pip install -e .[rl]    # Stable-Baselines3 and TensorBoard
```

## Run Tests

```bash
python3 -m pytest
```

The suite passes without Chrono, OSQP, Stable-Baselines3, or external track
data.

## Run The MPC Lap With Mock Backend

```bash
python3 scripts/run_mpc_lap.py --config configs/experiments/mpc_yas_marina_flat.yaml --backend mock
```

The default Yas Marina track comes from TUMFTM `tracks/YasMarina.csv` with
`racelines/YasMarina.csv` as the optional raceline. Logs and metrics are written
under `logs/`.

## Run With PyChrono

Install PyChrono in a Python 3.12 conda environment, then run:

```bash
python3 scripts/check_chrono_backend.py
python3 scripts/run_mpc_lap.py --config configs/experiments/mpc_yas_marina_flat.yaml --backend chrono
```

The first Chrono mode is `PyChronoKinematicBackend`: it creates a Chrono system,
flat ground, and chassis body while preserving the scaffold kinematic vehicle
model. This proves the direct Chrono training path and keeps the controller/RL
interfaces stable. A full `pychrono.vehicle` model is the next fidelity upgrade.

## Process Yas Marina Track Data

Do not vendor the full TUMFTM racetrack-database into this repo. The small
processed Yas Marina artifacts are stored under `tracks/yas_marina/processed/`.

```bash
git clone https://github.com/TUMFTM/racetrack-database /path/to/racetrack-database
python3 scripts/process_track.py --tumftm-root /path/to/racetrack-database
python3 scripts/plot_track.py
```

The flexible CSV loader infers common centerline, raceline, and width columns.
If your source file uses unusual column names, normalize it to:

```text
x_m,y_m,w_tr_right_m,w_tr_left_m
```

## Level-1 Track Limits And Curbs

Track limits use TUMFTM left/right widths. Level-1 curbs are flat semantic
1-meter edge zones along both sides of the full loop. They do not alter physics,
but MPC/RL logs and metrics report curb usage and RL rewards can penalize it.

## Chrono Backend Later

Training code should continue to call:

```python
ChronoDirectBackend.reset()
ChronoDirectBackend.step(command, dt)
ChronoDirectBackend.get_state()
```

Real Project Chrono initialization, vehicle spawning, state extraction, and
command application belong only in `src/chrono_a2rl/chrono_interface/`.

## Train First RL Speed Policy

The first RL policy does not output steering, throttle, or brake. It outputs a
single `target_speed_scale`, while MPC and PID remain responsible for low-level
control.

```bash
python3 scripts/train_rl_speed_policy.py --config configs/experiments/rl_speed_policy_yas_marina.yaml
```

If Stable-Baselines3 is missing, the script raises an actionable install
message.

## Current Limitations

- Full `pychrono.vehicle` dynamics are not implemented yet; the first Chrono
  mode uses an A2RL-style kinematic chassis body.
- Vehicle dynamics parameters are approximate public-reference research values,
  not private A2RL team data.
- Curbs are level-1 semantic flat edge zones; they do not alter mock or Chrono
  tire contact yet.
- The first lateral MPC can use OSQP, but the Yas Marina MVP config uses the
  deterministic proportional fallback by default until the QP model is tuned for
  the real track.
- ROS 2 runtime integration is documented but intentionally not in the training
  loop.

## Roadmap

1. Replace the kinematic Chrono chassis with a full `pychrono.vehicle` EAV-style model.
2. Validate Yas Marina TUMFTM processing against known lap geometry.
3. Upgrade lateral MPC to CasADi/acados or a fuller OSQP model.
4. Add physical curb and friction zones.
5. Train and evaluate the high-level SB3 speed policy.
6. Add ROS 2 runtime bridge after the direct training loop is stable.
