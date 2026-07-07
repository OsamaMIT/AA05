# Track Model

## TUMFTM Data

The loader accepts TUMFTM racetrack-database style CSV files and infers common
column names for:

- centerline coordinates
- optional raceline coordinates
- left and right track widths

The repository does not vendor the external dataset. Use
`scripts/download_track_data.py` for instructions and `scripts/process_track.py`
to regenerate:

```text
tracks/yas_marina/processed/yas_marina.csv
tracks/yas_marina/processed/yas_marina_raceline.csv
tracks/yas_marina/processed/yas_marina_metadata.yaml
```

The committed Yas Marina processed files are derived from
`TUMFTM/racetrack-database` commit
`e59595d1f3573b30d1ded6a08984935b957688e0`, using
`tracks/YasMarina.csv` and `racelines/YasMarina.csv`.

## Geometry

`TrackGeometry` stores a closed loop without a duplicated final point. It
computes:

- arc length `s`
- segment heading
- signed curvature
- closed-loop interpolation
- Frenet projection from `(x, y)` to `(s, n)`
- left/right boundary distances

Positive `n` means the vehicle is left of the centerline relative to travel
direction.

## Fallback Track

When Yas Marina data is absent, the config enables a synthetic oval. This keeps
unit tests, controller development, and RL plumbing runnable on any machine.

## Curbs

Curbs are loaded from `curbs.yaml` as semantic intervals:

```text
side, s_start, s_end, width, height, friction, type, penalty_weight, legal_status
```

Level 0 disables curbs. Level 1 marks semantic curbs only. Physical height,
friction, and tire interaction are deferred to later Chrono terrain work.

For the level-1 Yas Marina model, curbs are full-loop flat edge zones:

- left curb: `width_left - curb_width <= n <= width_left`
- right curb: `-width_right <= n <= -width_right + curb_width`
- default width: `1.0 m`
- default penalty weight: `0.2`

Track limits remain the TUMFTM boundaries. Curb usage is legal but penalized;
outside the boundaries is off-track.
