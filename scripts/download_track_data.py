#!/usr/bin/env python3
"""Help obtain TUMFTM racetrack-database without vendoring it."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/TUMFTM/racetrack-database"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clone-to",
        type=Path,
        help="Optional destination where the external database should be cloned.",
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Only print instructions.",
    )
    args = parser.parse_args()

    print("External track source:")
    print(f"  {REPO_URL}")
    print()
    print("Recommended workflow:")
    print("  git clone https://github.com/TUMFTM/racetrack-database /path/to/racetrack-database")
    print("  python3 scripts/process_track.py --input /path/to/racetrack-database/<yas-marina-csv>")
    print()
    print("Large external datasets are intentionally not vendored into this repo.")

    if args.clone_to and not args.no_clone:
        subprocess.run(["git", "clone", REPO_URL, str(args.clone_to)], check=True)


if __name__ == "__main__":
    main()
