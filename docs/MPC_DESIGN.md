# MPC Design

## Purpose

MPC is the first controller because it gives a deterministic, inspectable
baseline before adding RL. It also creates a safer interface for RL: the first
policy only scales target speed, while MPC handles lateral tracking.

## Inputs

- `VehicleState`: speed, yaw rate, steering angle, actuator feedback.
- `TrackState`: lateral error `n`, heading error, curvature, boundaries.
- `ControllerReference`: target speed, lookahead pose, target curvature.

## Outputs

`LateralMPCController` outputs steering only. `SpeedPIDController` outputs
throttle/brake. The lap runner combines those into a DBW `VehicleCommand`, then
passes it through the safety supervisor and DBW model.

## Model

The first MPC uses a small linearized bicycle lateral model:

```text
e_y[k+1]   = e_y[k] + v * dt * e_psi[k]
e_psi[k+1] = e_psi[k] + v / L * dt * delta[k] - v * kappa * dt
```

It penalizes lateral error, heading error, steering magnitude, and steering
rate. It constrains steering angle and steering rate.

## Solver Strategy

If OSQP is enabled and installed, a compact QP is solved over the steering
sequence. The Yas Marina MVP config disables OSQP by default and uses the
deterministic proportional fallback with curvature feedforward because that path
currently completes the conservative real-track lap reliably. The public
interface is identical so the solver can later be replaced or retuned with
CasADi/acados.
