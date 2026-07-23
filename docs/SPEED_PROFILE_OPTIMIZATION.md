# Speed Profile Optimization

## Purpose

`aa optimize` finds a faster profile that the current deterministic controller
can repeatedly complete. It does not train PPO and does not claim to identify
the physical limit of the real EAV24/EAV25 chassis.

The current Chrono mode remains kinematic. The result is therefore the
repeatable limit of the configured raceline, lateral controller, PID, DBW
model, and kinematic backend. The same optimizer can later evaluate a full
`pychrono.vehicle` model without changing its public workflow.

## Search Loop

The optimizer:

1. Generates or loads the configured baseline profile.
2. Runs a flying lap with the speed PID and a zero PPO pedal residual.
3. Divides the closed loop into configurable spatial segments.
4. Marks segments healthy using boundary margin, raceline error, heading error,
   and steering saturation.
5. Raises healthy segment targets by a small percentage.
6. Reapplies forward acceleration and backward braking feasibility passes.
7. Accepts only a faster completed lap within hard tracking limits.
8. Locks neighborhoods around failures and continues searching elsewhere.
9. Stops at the iteration limit, scale limit, or convergence.
10. Applies a safety margin and validates the output with one final lap.

The default search uses `50 m` segments, 1% increments, an upper scale of
`1.20`, up to 24 iterations, and a 1% final margin. The finer segmentation is
important in the short, shallow bends after Turn 2: those sections can be
raised without also raising the next substantial braking zone.

## Coast Through Shallow Bends

The speed PID has an asymmetric coast window. It coasts when the actual speed
is no more than `5.4 km/h` above a shallow profile dip, rather than applying
brake for every small negative speed error. The integral term decays while
coasting, and braking resumes normally when the target drop exceeds that
window. The PPO pedal residual is suppressed during explicit coast mode.

This preserves the profile as a feasibility target while allowing speed to
carry through small bends. It does not suppress braking for a real low-speed
corner.

The generated Yas Marina baseline complements this with a smooth,
curvature-dependent lateral acceleration envelope. It allows up to `27 m/s²`
in very shallow bends, blends back to `19 m/s²`, and reaches the normal limit
at `0.012 rad/m`. This raises flowing sections such as the bends after Turn 2
without changing the cap in substantial corners.

## Run

```bash
aa optimize --backend chrono
```

Useful overrides:

```bash
aa optimize --backend mock --iterations 3
aa optimize --backend chrono --iterations 12 \
  --output-dir artifacts/profile_optimization/long_search
```

The command reports all speeds in km/h and lap times in `M:SS.mmm`.

## Artifacts

Each run writes:

- `optimized_speed_profile.csv`: `s_m`, baseline speed, optimized speed, and
  final scale.
- `iteration_history.csv`: candidate decision and whole-lap diagnostics.
- `segment_diagnostics.csv`: final spatial margin, tracking, saturation, speed,
  coast and braking statistics, with baseline columns for comparison.
- `optimization_summary.yaml`: headline timing and validation results.

Generated runs are intentionally ignored by Git.

To activate one:

```yaml
speed_profile:
  profile_path: artifacts/profile_optimization/RUN_ID/optimized_speed_profile.csv
```

`generate_speed_profile` will then load and closed-loop interpolate the artifact
while retaining the configured minimum and 300 km/h maximum.

## Acceptance

A faster candidate is rejected if it:

- fails to complete the full start/finish crossing
- violates the hard boundary margin
- exceeds global RMS or peak raceline-error limits
- exceeds heading-error limits
- introduces excessive steering saturation
- does not improve the accepted lap time

The optimizer never rewards a crash for reaching a segment quickly. Only a
completed lap can replace the accepted profile.

## Full Vehicle Upgrade

With a full Chrono vehicle, replace the scalar curvature cap with a
speed-dependent G-G-V envelope derived from:

- tire combined-slip behavior
- aerodynamic downforce and drag
- load transfer
- power and gearing
- braking capacity

The optimizer's completed-lap validation and segment search remain useful, but
the initial profile will then represent physical tire and power limits rather
than constant acceleration assumptions.
