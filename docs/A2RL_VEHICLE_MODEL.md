# A2RL-Style Vehicle Model

## Scope

This repository models an A2RL-style Dallara EAV24 / Super Formula SF23. It is
not an exact EAV24 replica. Public sources do not provide complete tire curves,
aerodynamic maps, suspension geometry, inertia, gear ratios, or actuator
transfer functions.

Every value-bearing vehicle parameter is labeled:

- `public`: stated in public A2RL material.
- `proxy_sf19_sf23`: public Super Formula information used as a proxy.
- `estimate`: a physically plausible starting point requiring calibration.
- `tunable`: an intentional simulation or environment control.

The source files live in `vehicles/a2rl_style_eav24/`. The typed loader rejects
unknown provenance labels.

## Public And Proxy Basis

The model encodes a 690 kg autonomized Dallara SF23/EAV24-style platform, a
turbocharged 2.0 L inline-four near 410 kW, rear-wheel drive, a six-speed 3MO
gearbox, Brembo carbon brakes with electro-hydraulic actuation, and Yokohama
Advan racing slicks. The autonomous stack records seven Sony IMX728 cameras,
four ZF ProWave radars, three Seyond Falcon Kinetic lidars, GPS, and a Neousys
RGS-8805GC computer.

SF19/SF23 proxy dimensions are 5.233 m long, 1.910 m wide, 0.960 m high, with
a 3.115 m wheelbase. Track widths, CG, inertia, aero, tire stiffness, brake
capacity, suspension rates, and actuator dynamics are estimates.

## Model Levels

| Level | Implementation | Purpose |
|---|---|---|
| 0 | Existing `MockChronoBackend` / `PyChronoKinematicBackend` | Fast historical comparison |
| 1 | Constrained kinematic bicycle | Aero and finite actuator response |
| 2 | Dynamic bicycle | Default physical research model |
| 3 | PyChrono rigid force body | Chrono chassis representation using Level 2 force resolution |
| 4 | Full multibody suspension | Future work |

Level 3 does not claim to be a complete `pychrono.vehicle` suspension. It uses
the tested Level 2 force model and mirrors the resulting state into an
approximate Chrono rigid body. A real Level 4 implementation needs measured
hardpoints, compliant elements, wheel inertias, and validated tire data.

## Running

Run the dynamic experiment:

```bash
python3 scripts/run_mpc_lap.py \
  --config configs/experiments/a2rl_dynamic_vehicle_yas_marina.yaml
```

Run calibration tests and generate plots:

```bash
python3 scripts/validate_a2rl_vehicle_model.py
```

The dynamic model intentionally exposes limitations in a controller tuned for
the old kinematic car. A failed dynamic lap is not corrected by removing tire
saturation; steering constraints and MPC/MPCC tuning must be updated.

## Comparison Warning

Lap times must not be compared with A2RL benchmarks unless the track layout,
timing line, tire state, aero map, power limits, speed restrictions, and timing
method are verified. A kinematic model can appear deceptively competitive
because it lacks combined slip, aero drag, yaw transients, and actuator delay.

