# MPC Design

## Control Split

The default training controller is tube-aware pure pursuit. It follows the
TUMFTM raceline cheaply while PPO explores throttle and braking. The full
linear Tube MPC implementation is retained in
`configs/controller/mpc_lateral_full_tube.yaml` for later fine-tuning.

## Prediction Model

The nominal three-state model is:

```text
e_y[k+1]     = e_y[k] + v dt e_psi[k]
e_psi[k+1]   = e_psi[k] + v/L dt delta[k] - v kappa[k] dt
delta[k+1]   = (1-alpha) delta[k] + alpha delta_cmd[k]
```

`e_y` and `e_psi` are errors relative to the centerline or TUMFTM raceline,
`delta` is actuator steering, and `alpha` models steering lag. Reference
generation samples path curvature at every prediction step, allowing the QP to
anticipate curvature changes rather than repeating the current value.

OSQP minimizes lateral error, heading error, steering state, steering command,
and steering-command rate. Hard constraints cover steering magnitude,
steering rate, heading error, steering state, and the left/right track corridor.

## Robust Tube

The actual state is represented as a nominal state plus bounded error:

```text
x = z + e
u = v - K e
```

A discrete LQR gain `K` stabilizes the ancillary error dynamics. Bounded
lateral, heading, and steering disturbances are propagated through the
closed-loop model to form an iterated robust invariant zonotope. Its
axis-aligned state and input bounds tighten:

```text
track boundaries
heading limit
steering-state limit
steering-command limit
```

The ancillary correction is limited by the computed tube input bound, so the
applied command remains inside the input tightening used by the nominal QP.
Nominal and ancillary weights are independently configurable.

## Actuator Consistency

DBW first applies steering lag and then enforces the configured physical output
rate. The Tube prediction model uses the same lag time constant. This ordering
is important: rate-limiting the target before lag would silently reduce the
physical steering rate and invalidate the prediction model.

## Failure Behavior

The fast profile uses `mode: pure_pursuit` and does not instantiate OSQP. The
fine-tuning profile uses `mode: full_tube` and `use_osqp: true`. If its QP is
infeasible, the controller logs a warning and uses pure pursuit for that
control step. Solver status, nominal steering, ancillary correction, and tube
bounds are included in evaluation logs.

The validated mock-backend baseline completed Yas Marina with every QP solved.
The current model remains kinematic and linear; nonlinear tire forces, load
transfer, aero balance, and combined-slip limits are future fidelity work.
