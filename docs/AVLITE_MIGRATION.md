# AVLite Migration

## Target

This repository targets [AVLite](https://github.com/AV-Lab/avlite) `0.5.3` and
the `0.5.x` API line. The migration
was developed against upstream commit
`a974bcef5fd72ff8875f61f5f51350e4f1957ce3`.

AVLite is now a required dependency. The former project modules remain as
domain libraries, while AVLite owns their composition through standard
strategy and bridge interfaces.

## Plugin

`src/avlite_a2rl` is a normal AVLite community plugin. Its thin subclasses are
important: they register classes inside AVLite's plugin namespace while the
implementations remain importable and unit-testable in
`chrono_a2rl.avlite_stack`.

The runtime sets these paths automatically:

```text
AVLITE_CONFIG_DIR=<repo>/avlite_profiles
AVLITE_DATA_DIR=<repo>/avlite_data
AVLITE_PLUGINS_DIR=<repo>/src
```

No files are copied into AVLite's installation directory or the user's home
configuration.

## Track Conversion

`scripts/export_avlite_assets.py` converts the processed TUMFTM Yas Marina
geometry into:

```text
GlobalPlan: ReferenceLine, ReferenceSpeed, LeftBound, RightBound
RaceMap:    LeftBound, RightBound, ReferencePoint
```

The reference line follows the TUMFTM raceline. Bounds preserve the original
left/right widths after accounting for raceline lateral offset. The manifest
records the source experiment and generated speed range.

## Compatibility

The Gym observation and action arrays are unchanged. Existing compatible SB3
models can still be evaluated, but all simulator access now crosses
`A2RLChronoWorldBridge`. Set `avlite.enabled: false` only for explicit legacy
comparison; new experiment configs enable AVLite.

The internal `ChronoDirectBackend` has not been removed because it is the plant
facade used by the bridge. New application code should use
`create_simulation_backend`, `AVLiteBackendAdapter`, or AVLite's
`WorldBridge` interface instead of constructing it directly.

## Validation

```bash
aa stack --validate --backend mock
python3 -m pytest
python3 scripts/run_mpc_lap.py --backend mock
```

The first command loads the real AVLite profile, community plugin, map, global
plan, factory, executor, planner, controller, task, and world bridge, then
executes one synchronous simulation step.
