#!/usr/bin/env python3
"""Process a TUMFTM-style CSV into the local Yas Marina processed path."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chrono_a2rl.track.track_loader import load_tumftm_csv


DEFAULT_SOURCE_COMMIT = "e59595d1f3573b30d1ded6a08984935b957688e0"
UPSTREAM_URL = "https://github.com/TUMFTM/racetrack-database"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input TUMFTM-style track CSV path.")
    parser.add_argument(
        "--raceline-input",
        help="Optional TUMFTM-style raceline CSV path.",
    )
    parser.add_argument(
        "--tumftm-root",
        type=Path,
        help="Path to a local TUMFTM/racetrack-database clone.",
    )
    parser.add_argument(
        "--output",
        default="tracks/yas_marina/processed/yas_marina.csv",
        help="Processed CSV output path.",
    )
    parser.add_argument(
        "--raceline-output",
        default="tracks/yas_marina/processed/yas_marina_raceline.csv",
        help="Processed raceline CSV output path.",
    )
    parser.add_argument(
        "--metadata-output",
        default="tracks/yas_marina/processed/yas_marina_metadata.yaml",
        help="Processed source metadata YAML path.",
    )
    parser.add_argument(
        "--curbs-output",
        default="tracks/yas_marina/curbs.yaml",
        help="Level-1 curb YAML output path.",
    )
    parser.add_argument("--curb-width", type=float, default=1.0)
    parser.add_argument("--curb-penalty-weight", type=float, default=0.2)
    args = parser.parse_args()

    input_path, raceline_path = _resolve_source_paths(args.input, args.raceline_input, args.tumftm_root)
    track = load_tumftm_csv(
        input_path,
        name="yas_marina",
        raceline_path=raceline_path,
    )
    arrays = track.sample_arrays()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "x_m": arrays["x"],
            "y_m": arrays["y"],
            "w_tr_right_m": arrays["width_right"],
            "w_tr_left_m": arrays["width_left"],
            "s_m": arrays["s"],
            "heading_rad": arrays["heading"],
            "curvature_radpm": arrays["curvature"],
        }
    ).to_csv(output, index=False)

    raceline_output = ROOT / args.raceline_output
    if track.raceline is not None:
        raceline_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x_m": track.raceline[:, 0], "y_m": track.raceline[:, 1]}).to_csv(
            raceline_output,
            index=False,
        )

    metadata_output = ROOT / args.metadata_output
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = _make_metadata(input_path, raceline_path, args.tumftm_root, track)
    metadata_output.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    curbs_output = ROOT / args.curbs_output
    curbs_output.parent.mkdir(parents=True, exist_ok=True)
    curbs_output.write_text(
        yaml.safe_dump(
            _make_level1_curbs(track.length, args.curb_width, args.curb_penalty_weight),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote processed track: {output}")
    if track.raceline is not None:
        print(f"Wrote processed raceline: {raceline_output}")
    print(f"Wrote metadata: {metadata_output}")
    print(f"Wrote level-1 curbs: {curbs_output}")
    print(f"Track length: {track.length:.2f} m")


def _resolve_source_paths(
    input_arg: str | None,
    raceline_arg: str | None,
    tumftm_root: Path | None,
) -> tuple[Path, Path | None]:
    if tumftm_root is not None:
        track_path = tumftm_root / "tracks" / "YasMarina.csv"
        raceline_path = tumftm_root / "racelines" / "YasMarina.csv"
        if not track_path.exists():
            raise FileNotFoundError(f"Missing TUMFTM track CSV: {track_path}")
        if not raceline_path.exists():
            raceline_path = None
        return track_path, raceline_path
    if input_arg is None:
        raise ValueError("Provide --input or --tumftm-root")
    return Path(input_arg), Path(raceline_arg) if raceline_arg else None


def _make_metadata(
    track_path: Path,
    raceline_path: Path | None,
    tumftm_root: Path | None,
    track,
) -> dict:
    commit = _git_commit(tumftm_root) if tumftm_root is not None else DEFAULT_SOURCE_COMMIT
    return {
        "source": {
            "repository": UPSTREAM_URL,
            "commit": commit or DEFAULT_SOURCE_COMMIT,
            "license": "LGPL-3.0; retain upstream attribution when redistributing derived data.",
            "track_path": _source_path(track_path, tumftm_root),
            "raceline_path": _source_path(raceline_path, tumftm_root) if raceline_path else "",
        },
        "processed": {
            "track_name": "yas_marina",
            "track_length_m": float(track.length),
            "centerline_points": int(len(track.centerline)),
            "raceline_points": int(len(track.raceline)) if track.raceline is not None else 0,
            "width_left_min_m": float(track.width_left.min()),
            "width_left_max_m": float(track.width_left.max()),
            "width_right_min_m": float(track.width_right.min()),
            "width_right_max_m": float(track.width_right.max()),
        },
        "level_1_curbs": {
            "model": "flat_semantic_full_edge_zones",
            "physics_height_m": 0.0,
            "friction_multiplier": 1.0,
        },
    }


def _source_path(path: Path | None, tumftm_root: Path | None) -> str:
    if path is None:
        return ""
    if tumftm_root is None:
        return str(path)
    try:
        return str(path.relative_to(tumftm_root))
    except ValueError:
        return str(path)


def _git_commit(tumftm_root: Path | None) -> str | None:
    if tumftm_root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(tumftm_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def _make_level1_curbs(track_length: float, curb_width: float, penalty_weight: float) -> dict:
    common = {
        "s_start": 0.0,
        "s_end": float(track_length),
        "width": float(curb_width),
        "height": 0.0,
        "friction": 1.0,
        "type": "flat_semantic_edge_zone",
        "penalty_weight": float(penalty_weight),
        "legal_status": "legal_but_penalized",
    }
    return {
        "curb_level": 1,
        "description": "Level-1 flat curbs: semantic full-loop edge zones only; track limits remain TUMFTM widths.",
        "curbs": [
            {"side": "left", **common},
            {"side": "right", **common},
        ],
    }


if __name__ == "__main__":
    main()
