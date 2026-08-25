# AVLite A2RL Plugin

This community plugin registers the repository's Chrono world bridge, TUMFTM
race planner, MPC/PID controller, and lap telemetry task with AVLite. The
implementation remains in `chrono_a2rl.avlite_stack`; these thin subclasses
give AVLite ownership of plugin discovery and strategy registration.

The repository's `aa stack` commands set `AVLITE_PLUGINS_DIR` automatically.
