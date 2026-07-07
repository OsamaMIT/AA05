#!/usr/bin/env python3
"""Plot the configured track, falling back to the synthetic track if needed."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "logs" / ".matplotlib"))

import matplotlib.pyplot as plt

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chrono_a2rl.common.config import load_yaml
from chrono_a2rl.track.track_loader import load_track_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/track/yas_marina.yaml")
    parser.add_argument("--output", default="logs/track_preview.png")
    args = parser.parse_args()

    config = load_yaml(args.config)
    track = load_track_from_config(config)
    arrays = track.sample_arrays()
    plt.figure(figsize=(8, 6))
    plt.plot(arrays["x"], arrays["y"], label=track.name)
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
