#!/usr/bin/env python3
"""Export the repository's TUMFTM track into AVLite race-map formats."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.track.speed_profile import generate_speed_profile
from chrono_a2rl.track.track_loader import load_track_from_config


DEFAULT_REFERENCE_POINT = [24.46992202098782, 54.60522506805341, 0.0]


def export_assets(
    config_path: str | Path,
    output_dir: str | Path = "avlite_data",
) -> dict[str, str | int | float]:
    """Write an AVLite GlobalPlan, RaceMap, and provenance manifest."""

    config = load_experiment_config(config_path)
    track = load_track_from_config(config["track"])
    profile = generate_speed_profile(track, config.get("speed_profile", {}))
    output = Path(output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    reference_line: list[list[float]] = []
    left_d: list[float] = []
    right_d: list[float] = []
    left_bound: list[list[float]] = []
    right_bound: list[list[float]] = []
    for s in track.s_nodes:
        sample = track.interpolate(float(s))
        race_x, race_y = track.raceline_xy_at(float(s))
        race_n = track.raceline_lateral_offset_at(float(s))
        normal_x = -math.sin(sample.heading)
        normal_y = math.cos(sample.heading)
        reference_line.append([race_x, race_y, 0.0])
        left_d.append(sample.width_left - race_n)
        right_d.append(-sample.width_right - race_n)
        left_bound.append(
            [
                sample.x + sample.width_left * normal_x,
                sample.y + sample.width_left * normal_y,
                0.0,
            ]
        )
        right_bound.append(
            [
                sample.x - sample.width_right * normal_x,
                sample.y - sample.width_right * normal_y,
                0.0,
            ]
        )

    plan_path = output / "yas_marina_a2rl_global_plan.json"
    map_path = output / "yas_marina_a2rl.map.json"
    manifest_path = output / "yas_marina_a2rl_manifest.yaml"
    plan_path.write_text(
        json.dumps(
            {
                "ReferenceLine": reference_line,
                "ReferenceSpeed": [float(value) for value in profile.speed],
                "LeftBound": left_d,
                "RightBound": right_d,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    map_path.write_text(
        json.dumps(
            {
                "LeftBound": left_bound,
                "RightBound": right_bound,
                "ReferencePoint": DEFAULT_REFERENCE_POINT,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    experiment_path = Path(config["experiment_path"])
    try:
        experiment_name = experiment_path.relative_to(ROOT).as_posix()
    except ValueError:
        experiment_name = str(experiment_path)
    manifest = {
        "format": "AVLite GlobalPlan and RaceMap",
        "avlite_api_target": "0.5.3",
        "source_experiment": experiment_name,
        "source_track": "TUMFTM/racetrack-database YasMarina",
        "source_metadata": config["track"].get("metadata_path", ""),
        "track_length_m": float(track.length),
        "reference_points": len(reference_line),
        "speed_profile_min_kmh": float(min(profile.speed) * 3.6),
        "speed_profile_max_kmh": float(max(profile.speed) * 3.6),
        "files": {
            "global_plan": plan_path.name,
            "race_map": map_path.name,
        },
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return {
        "global_plan": str(plan_path),
        "race_map": str(map_path),
        "manifest": str(manifest_path),
        "track_length_m": float(track.length),
        "points": len(reference_line),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/rl_planner_yas_marina.yaml",
        help="Experiment whose raceline and speed profile should be exported.",
    )
    parser.add_argument("--output-dir", default="avlite_data")
    args = parser.parse_args()
    for key, value in export_assets(args.config, args.output_dir).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
