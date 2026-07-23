# EAV24 Approximation Calibration Notes

This is an A2RL-style research model, not an exact EAV24 replica. Public
sources identify the platform, mass, powertrain family, brakes, tires, sensor
stack, and nominal top speed, but do not publish tire curves, aero maps,
suspension rates, inertia tensors, gear ratios, or actuator transfer functions.

Every scalar carrying a `value` in this directory also carries one of:
`public`, `proxy_sf19_sf23`, `estimate`, or `tunable`. Estimates are initial
conditions for calibration, not measured vehicle data.

Calibrate in this order:

1. Match top speed with `cda_total` and `max_power_kw`.
2. Match low-speed cornering with tire friction and cornering stiffness.
3. Match high-speed cornering with `cla_total` and aero balance.
4. Match braking distance with tire friction, brake force, and aero load.
5. Match yaw transients with yaw inertia, axle stiffness, and steering lag.

Do not compare simulated lap times with A2RL results until track geometry,
timing line, tire state, aero, power limits, and control constraints are known
to be comparable.
