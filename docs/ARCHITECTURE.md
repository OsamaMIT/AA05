# Architecture

AVLite owns stack assembly, execution contracts, plugin discovery, maps, plans,
and world integration. Racing-specific physics and algorithms remain small,
testable implementations behind those contracts.

```text
AVLite SyncExecuter
  c20 A2RLGlobalRacePlanner -> AVLite GlobalPlan
  c20 A2RLReferenceLocalPlanner -> AVLite LocalPlan
  c30 A2RLMPCControlStrategy -> A2RLAckermannCommand
  c40 A2RLChronoWorldBridge -> Chrono/mock plant
  c40 A2RLLapTelemetryTask -> CSV and metrics
```

The portable assembly is defined by:

```text
avlite_profiles/a2rl_chrono_yas_marina.yaml
avlite_data/yas_marina_a2rl_global_plan.json
avlite_data/yas_marina_a2rl.map.json
src/avlite_a2rl/
```

## Layer Mapping

| Racing concern | AVLite owner | Repository implementation |
| --- | --- | --- |
| Track map and limits | `Map` / `RaceMap` | TUMFTM asset exporter |
| Geometric route and speed | `GlobalPlannerStrategy` | `A2RLGlobalRacePlanner` |
| Active reference | `LocalPlanningStrategy` | `A2RLReferenceLocalPlanner` |
| Steering and speed tracking | `ControlStrategy` | `A2RLMPCControlStrategy` |
| Simulation and sensors | `WorldBridge` | `A2RLChronoWorldBridge` |
| Lap supervision and logs | `TaskStrategy` | `A2RLLapTelemetryTask` |
| PPO rollouts | Gymnasium application adapter | `AVLiteBackendAdapter` |

## Training Path

SB3 needs many independent, synchronous Gymnasium environments. Each worker
therefore uses a lightweight backend-shaped adapter over the same AVLite
`WorldBridge` rather than constructing the graphical app or ROS transport:

```text
Stable-Baselines3 PPO
  -> ChronoRacingPlannerEnv
  -> MPC/PID + safety supervisor
  -> AVLiteBackendAdapter
  -> A2RLChronoWorldBridge
  -> ChronoDirectBackend internal plant
```

The adapter preserves the existing observation/action contract, checkpoint
compatibility, deterministic resets, and vectorized throughput. Chrono imports
remain inside `chrono_interface/`; application code depends on the AVLite
bridge contract.

## Execution Modes

```text
aa stack       complete AVLite SyncExecuter application
aa train       vectorized Gym application over AVLite WorldBridge
aa eval/watch  deterministic Gym application over AVLite WorldBridge
```

ROS is still absent from the training loop. A future ROS implementation is a
different `WorldBridge`, selected by an AVLite profile without changing the
planner, controller, or task contracts.

## Vehicle And Track Domains

`vehicle/` owns A2RL-style dynamics, tires, aero, braking, actuators, and
telemetry. `track/` owns TUMFTM parsing, Frenet geometry, speed profiles, and
semantic curbs. These are domain libraries used by AVLite strategies, not
parallel orchestration frameworks.

The public-reference A2RL EAV24/EAV25 parameters remain estimates where public
data is unavailable. AVLite changes software composition; it does not make
those estimates authoritative.
