# A2RL Vehicle Calibration

## Principle

Calibration should identify a parameter from the test that is sensitive to it.
Do not tune every parameter against lap time. Lap time mixes vehicle limits,
controller quality, track interpretation, and timing behavior.

## Recommended Sequence

1. **Top speed:** tune effective `CdA` and available power. Target about
   `300 km/h`, with a provisional tolerance of `10 km/h`.
2. **Low-speed lateral grip:** tune peak friction and cornering stiffness.
   Provisional target: `1.7-2.2 g`.
3. **High-speed cornering:** tune effective `ClA`, aero balance, and load
   sensitivity. Provisional target: `3.0-4.5 g`.
4. **Braking:** tune maximum brake request, friction, bias, and downforce.
   High-speed peak deceleration should exceed low-speed deceleration.
5. **Transients:** tune yaw inertia, axle stiffness, steering delay/rate, and
   response time against step-steer or measured telemetry.
6. **Controller:** only after open-loop vehicle tests pass, retune MPC/MPCC
   steering limits, prediction dynamics, and speed feasibility.

## Validation Command

```bash
python3 scripts/validate_a2rl_vehicle_model.py
```

Outputs under `logs/vehicle_validation/` include:

- acceleration and top-speed results
- `300->80 km/h` braking distance and peak deceleration
- constant-radius tire usage
- step-steering actuator and yaw response
- Level 0 versus Level 2 lap sanity summaries
- primitive and lap telemetry plots

Use `--skip-lap` for fast open-loop calibration.

## Acceptance Guidance

- Tire usage must remain at or below the configured ellipse safety factor.
- Steering must not jump to its target in one controller step.
- Full braking must reduce available lateral force.
- Top speed must result from power and drag, not only a hard clamp.
- Braking distance must be finite and aero-load dependent.
- State and yaw response must remain smooth under small inputs.

Public A2RL data is insufficient to claim benchmark accuracy. Record every
calibration dataset, parameter revision, weather condition, and model version.

