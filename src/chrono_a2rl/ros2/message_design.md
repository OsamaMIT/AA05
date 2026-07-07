# Future ROS 2 Message Design

## Simulator Update

```text
episode_id
update_id
sim_time
control_period
dt_since_last_update
vehicle_state
track_state
```

## Driver Command

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

## Timing Model

Training mode is synchronous direct Chrono stepping:

```text
policy/controller -> backend.step(command, dt) -> state
```

Runtime mode is event-driven ROS messaging:

```text
simulator update topic -> driver command topic -> simulator command gate
```

Commands carry source and target update labels so the runtime can reject stale
or misaligned commands.
