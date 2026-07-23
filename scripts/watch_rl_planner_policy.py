#!/usr/bin/env python3
"""Watch a trained PPO planner policy drive around the track in real time."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "logs" / ".matplotlib"))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chrono_a2rl.rl.visualize_policy import main


if __name__ == "__main__":
    main()
