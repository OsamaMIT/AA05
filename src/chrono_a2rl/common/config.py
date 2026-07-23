"""YAML configuration loading and experiment composition."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""

    yaml_path = resolve_path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {yaml_path}")
    return data


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve repository-relative, absolute, or base-relative paths."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if base_dir is not None:
        base_candidate = Path(base_dir) / candidate
        if base_candidate.exists():
            return base_candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""

    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config and merge referenced component configs.

    The experiment file may contain:
    component_configs:
      vehicle: configs/vehicle/a2rl_style_sf23.yaml
      track: configs/track/yas_marina.yaml
      controller:
        lateral: configs/controller/mpc_lateral.yaml
        speed: configs/controller/speed_pid.yaml
      simulation: configs/simulation/chrono_default.yaml
      rl: configs/rl/ppo_speed_policy.yaml

    The returned mapping has normalized top-level keys: vehicle, track,
    controller, simulation, rl, experiment.
    """

    experiment_path = resolve_path(path)
    experiment = load_yaml(experiment_path)
    component_refs = experiment.get("component_configs", {})

    merged: dict[str, Any] = {
        "vehicle": {},
        "track": {},
        "controller": {},
        "simulation": {},
        "rl": {},
        "experiment": {},
    }

    vehicle_ref = component_refs.get("vehicle")
    if vehicle_ref:
        merged["vehicle"] = load_yaml(resolve_path(vehicle_ref, experiment_path.parent))

    track_ref = component_refs.get("track")
    if track_ref:
        merged["track"] = load_yaml(resolve_path(track_ref, experiment_path.parent))

    simulation_ref = component_refs.get("simulation")
    if simulation_ref:
        merged["simulation"] = load_yaml(
            resolve_path(simulation_ref, experiment_path.parent)
        )

    controller_refs = component_refs.get("controller", {})
    if isinstance(controller_refs, dict):
        for name, ref in controller_refs.items():
            merged["controller"][name] = load_yaml(
                resolve_path(ref, experiment_path.parent)
            )
    elif controller_refs:
        merged["controller"] = load_yaml(resolve_path(controller_refs, experiment_path.parent))

    rl_ref = component_refs.get("rl")
    if rl_ref:
        merged["rl"] = load_yaml(resolve_path(rl_ref, experiment_path.parent))

    for key in ("vehicle", "track", "controller", "simulation", "rl"):
        if key in experiment:
            merged[key] = deep_merge(merged.get(key, {}), experiment[key])

    experiment_extras = {
        key: value
        for key, value in experiment.items()
        if key not in {"component_configs", "vehicle", "track", "controller", "simulation", "rl"}
    }
    merged["experiment"] = experiment_extras
    for key, value in experiment_extras.items():
        if key not in merged:
            merged[key] = deepcopy(value)
    merged["experiment_path"] = str(experiment_path)
    return merged
