# Product Requirements

## Goal

Build a staged autonomous racing training stack where an A2RL-style formula car
can complete stable autonomous laps on a flat Yas Marina track using an MPC
baseline before RL or ROS integration becomes critical.

## Users

- Researchers iterating on control, speed planning, and RL policies.
- Vehicle dynamics engineers preparing for real Chrono integration.
- Robotics engineers planning later ROS 2 runtime deployment.

## MVP Success

- Repository structure is modular and runnable.
- Config loading composes experiment, vehicle, track, controller, simulation,
  and RL settings.
- Track loading supports TUMFTM-style CSVs and synthetic fallback.
- Mock backend supports reset and stepping without Chrono.
- PyChrono backend can initialize a Chrono system and step through the same
  direct backend API when PyChrono is installed.
- Vehicle profile is based on public A2RL EAV24/EAV25 information while keeping
  private or unpublished values as replaceable estimates.
- MPC/PID lap runner completes a synthetic lap and saves metrics.
- Gymnasium environment exposes high-level speed scaling actions.
- Tests pass without external simulator dependencies.

## Non-Goals

- Private A2RL vehicle parameter reproduction.
- End-to-end RL steering/throttle/brake control.
- ROS 2 in the training loop.
- Physical curb modeling in the first milestone.
