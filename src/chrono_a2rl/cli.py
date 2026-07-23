"""Top-level terminal command dispatcher for the A2RL training stack."""

from __future__ import annotations

import sys
from collections.abc import Callable

from chrono_a2rl.evaluation.profile_optimizer import main as optimize_main
from chrono_a2rl.rl.evaluate_policy import main as eval_main
from chrono_a2rl.rl.train import main_planner, main_speed
from chrono_a2rl.rl.visualize_policy import main as watch_main


_COMMANDS: dict[str, tuple[str, Callable[[list[str] | None], None]]] = {
    "train": ("Train the bounded profile-speed pedal residual.", main_planner),
    "eval": ("Evaluate a saved policy and write metrics.", eval_main),
    "watch": ("Watch a saved policy in a live graphical view.", watch_main),
    "optimize": ("Find a faster repeatable PID speed profile.", optimize_main),
    "train-speed": ("Train the older speed-only policy.", main_speed),
}


def main(argv: list[str] | None = None) -> None:
    """Dispatch `aa <command>` to the requested workflow."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return

    command = args.pop(0)
    if command == "help":
        _dispatch_help(args)
        return

    if command not in _COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        _print_help(file=sys.stderr)
        raise SystemExit(2)

    _description, handler = _COMMANDS[command]
    old_argv = sys.argv
    sys.argv = [f"aa {command}", *args]
    try:
        handler(args)
    finally:
        sys.argv = old_argv


def _print_help(*, file=None) -> None:
    stream = file or sys.stdout
    print("usage: aa <command> [options]", file=stream)
    print("       aa help [command]\n", file=stream)
    print("Commands:", file=stream)
    print("  help         Show overview help or detailed help for a command.", file=stream)
    for name, (description, _handler) in _COMMANDS.items():
        print(f"  {name:<12} {description}", file=stream)
    print("\nExamples:", file=stream)
    print("  aa help", file=stream)
    print("  aa help train", file=stream)
    print("  aa train --total-timesteps 250000 --n-envs 4", file=stream)
    print("  aa train --resume latest", file=stream)
    print("  aa eval --model latest", file=stream)
    print("  aa watch --model latest", file=stream)
    print("  aa watch --camera follow --zoom-radius 120", file=stream)
    print("  aa optimize --backend chrono --iterations 8", file=stream)


def _dispatch_help(args: list[str]) -> None:
    if not args:
        _print_help()
        return

    command = args[0]
    if command not in _COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        _print_help(file=sys.stderr)
        raise SystemExit(2)

    _description, handler = _COMMANDS[command]
    old_argv = sys.argv
    sys.argv = [f"aa {command}", "--help"]
    try:
        try:
            handler(["--help"])
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main(sys.argv[1:])
