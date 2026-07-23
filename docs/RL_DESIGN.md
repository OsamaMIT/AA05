# RL Design

## Control Boundary

The default Stable-Baselines3 policy fine-tunes longitudinal control:

```text
action[0] = signed pedal residual in [-1, 1]

-1.0 -> maximum permitted brake residual
 0.0 -> no change to profile PID
+1.0 -> maximum permitted throttle residual
```

A speed PID is authoritative and converts the raceline speed profile into
throttle and brake. PPO can adjust that command by at most `8%`, and its
authority fades linearly to zero at `5 m/s` (`18 km/h`) profile error. This
prevents the learned policy from ignoring an accurate profile while preserving
room for small trail-braking and lap-time improvements. Throttle and brake
remain mutually exclusive and pass through DBW actuator filtering.

Steering is not in the action space. The TUMFTM green raceline is the fixed
lateral reference, and the fast default controller uses tube-aware pure pursuit
to produce every steering command:

```text
TUMFTM speed profile -> speed PID -> base throttle/brake
PPO -> bounded pedal residual
TUMFTM raceline -> lateral MPC -> steering
combined pedals + steering -> safety supervisor -> DBW -> Chrono/mock backend
```

The curvature speed profile is authoritative. PPO observes actual and target
speed and is rewarded for reducing their difference.

## Observation

The 33-value observation contains:

```text
vehicle speed
nominal local profile speed
lateral and heading errors
yaw rate and steering angle
current raceline curvature
left and right boundary distances
curb state and penalty weight
applied throttle and brake
previous signed pedal action
lap progress
fixed raceline offset
five future raceline curvatures
five future profile speeds
normalized track position
distance to shared frontier
corner phase
distance to apex
corner heading-completion fraction
geometry-derived trail-brake target
```

Future samples at `20, 40, 80, 120, 180 m` give the policy enough preview to
learn braking onset and release from the shape of the raceline. The final
observation is a moderate advisory brake reference calculated from the next
corner entry/apex, current speed, passive drag, and available braking distance.
It is never blended into the policy action.

## Reward

Validated progress is the main signal. The reward also includes:

```text
+ on-track progress
+ achieved-speed progress with line and boundary safety gates
+ profile-speed tracking accuracy
+ clean corner completion after exit clearance
+ frontier clearance and validated extension
- elapsed control steps
- raceline tracking error
- pedal/control effort and pedal changes
- curb use and sustained curb overuse
- instability
- off-track termination
- quadratic high-speed crash cost
- extra crash cost in a corner or exit-clearance zone
- corner-entry overspeed relative to the future raceline cap
+ controlled brake-assisted deceleration inside a required braking window
+ matching the geometry-derived trail-brake reference
- missing requested trail brake
- braking beyond the reference or configured limits
- excessive braking and corner underspeed
```

Requested target speed is not rewarded. The policy must actually accelerate and
remain on track. A corner reward is withheld until the car passes the apex,
traverses at least 90 percent of the segment, completes at least 80 percent of
the expected heading rotation, and remains on track for another `40 m`.

Braking shaping combines geometry and outcome. The reference activates within
`350 m` of an apex, reserves `45 m`, subtracts estimated drag and rolling
deceleration, and caps brake pressure at `0.70`. From corner entry to apex it
tapers with exponent `0.80`, retaining more brake deeper into entry before
reaching zero at and after the apex. Controlled
deceleration is rewarded only when brake is actually applied, so lifting and
aerodynamic drag cannot collect the braking bonus. PPO remains free to deviate
from the reference when another action produces better completed-lap reward.
With authoritative profile tracking enabled, these terms remain diagnostics;
the residual policy is trained on profile error and lap outcome instead of
being penalized for the PID's base brake command.

All positive per-step speed, line, and apex terms are gated by forward
displacement. A stationary car therefore receives only time, low-speed, and
stationary costs. Less than `0.01 m` progress per control step for `4 s`
terminates the episode as `stalled` and applies a `600` point penalty.

## Shared Frontier

Eight environments are assigned distinct roles by default:

```text
2 start-line cars
4 frontier-practice cars
2 randomized whole-track cars
```

Start-line cars validate genuine lap progress. Practice cars reset `150-250 m`
before the shared frontier, and random cars preserve coverage elsewhere. New
progress counts only after another `40 m` is completed on track. The callback
advances the frontier monotonically by at most `150 m` per update and stores its
state beside every checkpoint.

Reset speeds use `0.40-0.85` times the local raceline profile by default. This
keeps early pedal exploration conservative without removing the opportunity to
learn higher speeds through successful progress.

## PPO And Checkpoints

The one-dimensional action and revised observation semantics require fresh
training. Models are stored under:

```text
models/ppo_profile_speed_residual/<run_id>/
```

Do not load checkpoints from `models/ppo_planner_frontier`; those models expect
two actions with different meanings.

The earlier `ppo_longitudinal_progress`, `ppo_longitudinal_raceline`,
`ppo_longitudinal_full_tube`, and `ppo_longitudinal_fast_raceline` policies used
different controller, DBW, observation, or reward contracts. Start a fresh
profile-residual run. Fine-tuning can later load the resulting model while selecting
`configs/controller/mpc_lateral_full_tube.yaml`.

```bash
aa train
aa train --no-eval
aa train --total-timesteps 250000 --n-envs 4 --backend mock
aa train --longitudinal-action-deadband 0.08
aa train --resume latest
```

A plain `aa train` creates a fresh run and frontier. `--resume latest` and
`--resume-latest` restore the newest compatible model and its paired frontier.
Training automatically evaluates `final_model.zip` unless `--no-eval` is used.

Evaluation and graphical watch mode resolve the latest compatible run:

```bash
aa eval --model latest
aa watch --model latest --camera follow --zoom-radius 120
```

Evaluation CSV files include signed pedal action, requested/commanded/applied
throttle and brake, steering target, progress, corner diagnostics, and crash
costs. TensorBoard records the pedal action, actuator feedback, action changes,
frontier progress, and corner outcomes.

## Current Limitations

The mock and current Chrono scaffold backends use kinematic longitudinal
dynamics. The learned pedal timing must be retrained or fine-tuned when the
backend gains high-fidelity tire, aero, powertrain, and brake behavior. Lateral
MPC tuning also limits the maximum speed that can be held on the fixed raceline.
