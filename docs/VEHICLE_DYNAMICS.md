# Vehicle Dynamics

## State And Integration

The Level 2 model maintains planar body-frame longitudinal and lateral
velocities, world position, yaw, yaw rate, and actual actuator states. It uses
small configurable physics substeps inside each controller step.

The equations are:

```text
vx_dot = (Fx - Fy_front*sin(delta))/m + vy*yaw_rate
vy_dot = (Fy_rear + Fy_front*cos(delta))/m - vx*yaw_rate
yaw_rate_dot = (lf*Fy_front*cos(delta) - lr*Fy_rear)/Iz
```

World velocity is obtained by rotating body velocity through yaw. Low-speed
guards prevent division by zero but do not provide extra tire force.

## Aerodynamics

Drag and downforce are quadratic:

```text
drag = 0.5*rho*CdA*v^2
downforce = min(0.5*rho*ClA*v^2, configured clamp)
```

Downforce is split by aero balance and added to axle normal load. Drag is
subtracted from longitudinal force. `CdA` and `ClA` are effective areas, not
claimed official coefficients.

## Tires And Combined Slip

Front and rear slip angles drive linear axle lateral-force requests. Normal
load and load-sensitive friction define the available force. The selected
friction circle or ellipse scales longitudinal and lateral forces together:

```text
usage = sqrt((Fx/Fx_max)^2 + (Fy/Fy_max)^2)
```

The configured safety factor is `0.92`. Braking or accelerating therefore
reduces available cornering force. Full brake and full lateral force cannot
coexist.

## Load Transfer

Static weight distribution and aero define initial axle loads. Longitudinal
load transfer is:

```text
dFz = m*ax*h_cg/wheelbase
```

Acceleration moves load rearward and braking moves load forward. Approximate
lateral transfer indicators are computed from CG height, track width, and roll
stiffness distribution. A future four-wheel model should use these indicators
to resolve individual wheel loads.

## Powertrain And Brakes

The rear-drive force is the minimum of power-limited and rear-tire-limited
force. It includes drivetrain efficiency, a soft top-speed limiter, rolling
resistance, and aero drag. Brakes request axle forces from total deceleration
capacity and front bias; the tire ellipse limits the achieved force.

## Actuators

Steering, throttle, and brake each pass through clipping, transport delay,
first-order response, and rate limiting. Throttle/brake conflicts are resolved
before forces are computed. Gear shifts impose a finite torque interruption.

## Telemetry

Dynamic lap CSVs include acceleration, slip angles, axle tire usage, normal
loads, downforce, drag, drive/brake forces, command/actual actuator pairs, gear,
curb state, and track status.

