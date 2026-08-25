# ROS 2 Integration

ROS 2 is deferred until the AVLite/Chrono training stack is stable. It will be
implemented as an alternate AVLite `WorldBridge`, not inserted between SB3 and
the synchronous Chrono bridge.

## Modes

```text
training mode = synchronous AVLite Chrono WorldBridge stepping
runtime mode = event-driven AVLite ROS WorldBridge with labeled updates
```

Training mode calls the in-process Chrono plant through AVLite's world contract
for deterministic reset/step loops and high-throughput data collection.
Runtime mode will publish simulator updates and receive driver commands through
ROS topics or services behind a different bridge implementation.

## Future Simulator Update

Each simulator update should include:

```text
episode_id
update_id
sim_time
control_period
dt_since_last_update
vehicle_state
track_state
```

## Future Driver Command

Each driver command should include:

```text
episode_id
command_id
source_update_id
target_update_id
sim_time_observed
command_valid_from
command_valid_until
steering
throttle
brake
emergency_brake
```

The labeled IDs make delayed, dropped, or stale commands visible instead of
silently applying them to the wrong simulator update.
