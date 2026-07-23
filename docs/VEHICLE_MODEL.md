# Vehicle Model

## Public A2RL Basis

The repository vehicle profile is now based on the public A2RL autonomous car
description:

- Dallara EAV24 / EAV25 autonomous Super Formula platform
- EAV-25 is the Season 2 upgraded model
- 4 Piston Racing K20C1 engine, based on Honda 2.0 liter architecture
- Inline 4-cylinder, direct injection, turbocharged with intercooler
- 3MO 6-speed gearbox
- Front and rear pushrod suspension
- Brembo carbon discs and calipers, electro-hydraulically activated
- Yokohama Advan tires
- Sensor suite: 7 Sony IMX728 cameras, 4 ZF ProWave radars, 3 Seyond Falcon
  Kinetic FK1 lidars
- High-level computer: Neousys RGS-8805GC

Primary source: https://a2rl.io/autonomous-car-race
Published up-to-300 km/h platform specification:
https://www.aspireuae.ae/news/uaes-aspire-redefines-extreme-autonomous-sports-a2rl-unveils-autonomous-dallara-super-formula
A2RL reported 295 km/h autonomous speed source:
https://a2rl.io/news/42/295-kmph-without-human-driver-How-UAE-engineers-are-teaching-cars-to-race-themselves
EAV-25 upgrade source: https://a2rl.io/eav25

## Simulation Parameters

The current dynamics parameters are intentionally approximate. A2RL does not
publish exact mass, inertia tensor, aero maps, tire curves, actuator rate
limits, or braking maps. The MVP therefore uses conservative values that are
easy to replace:

- `300 km/h` published platform top-speed cap for race-speed training
  experiments; 295 km/h remains the reported achieved autonomous-speed
  reference
- estimated race-ready mass including autonomous stack
- Super Formula-scale wheelbase and body dimensions
- simple aggregate drag/rolling resistance model in the mock backend
- box-inertia estimate for the first PyChrono chassis body
- placeholder cornering stiffness and friction
- placeholder DBW actuator lags and steering-rate limits

The mock backend's `drag_coefficient` is an aggregate speed-squared loss in
`m/s^2`, not a physical aerodynamic drag coefficient.

## Chrono Mode

`PyChronoKinematicBackend` creates a Chrono system, fixed ground, and chassis
body using the A2RL-style mass/dimensions. It mirrors the kinematic state into
Chrono each step. This confirms the direct Chrono path while preserving the
controller and RL interfaces.

The next fidelity upgrade is to replace this internal chassis with a full
`pychrono.vehicle` model using the same public A2RL profile as the starting
point.
