"""Load and validate provenance-labeled A2RL vehicle configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chrono_a2rl.common.config import REPO_ROOT, deep_merge, load_yaml, resolve_path


ALLOWED_SOURCES = {"public", "proxy_sf19_sf23", "estimate", "tunable"}
CONFIG_FILES = (
    "vehicle_params.yaml",
    "mass_properties.yaml",
    "geometry.yaml",
    "aero.yaml",
    "tires.yaml",
    "brakes.yaml",
    "powertrain.yaml",
    "suspension.yaml",
    "actuators.yaml",
    "sensors.yaml",
)


@dataclass(frozen=True, slots=True)
class A2RLVehicleConfig:
    """Merged vehicle data with provenance validation and scalar accessors."""

    root: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path = "vehicles/a2rl_style_eav24") -> "A2RLVehicleConfig":
        directory = resolve_path(root)
        if not directory.is_dir():
            raise FileNotFoundError(f"A2RL vehicle model directory not found: {directory}")
        merged: dict[str, Any] = {}
        for name in CONFIG_FILES:
            path = directory / name
            if not path.exists():
                raise FileNotFoundError(f"Required A2RL vehicle config is missing: {path}")
            merged = deep_merge(merged, load_yaml(path))
        _validate_sources(merged)
        return cls(root=directory, data=merged)

    @classmethod
    def from_component_config(cls, config: dict[str, Any]) -> "A2RLVehicleConfig":
        return cls.load(config.get("model_root", "vehicles/a2rl_style_eav24"))

    def value(self, path: str, default: Any = None) -> Any:
        """Read a scalar from either a provenance mapping or a plain field."""

        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not None:
                    return default
                raise KeyError(f"Missing A2RL vehicle parameter: {path}")
            node = node[part]
        if isinstance(node, dict) and "value" in node:
            return node["value"]
        return node

    def source(self, path: str) -> str:
        node: Any = self.data
        for part in path.split("."):
            node = node[part]
        if not isinstance(node, dict) or "source" not in node:
            raise KeyError(f"Parameter does not carry provenance: {path}")
        return str(node["source"])

    def legacy_flat_config(self) -> dict[str, Any]:
        """Expose compatibility values consumed by existing controllers."""

        return {
            "mass": float(self.value("vehicle.mass.total_kg")),
            "wheelbase": float(self.value("geometry.wheelbase_m")),
            "front_axle_to_cg": float(self.value("geometry.cg_to_front_axle_m")),
            "rear_axle_to_cg": float(self.value("geometry.cg_to_rear_axle_m")),
            "body_length": float(self.value("geometry.length_m")),
            "body_width": float(self.value("geometry.width_m")),
            "body_height": float(self.value("geometry.height_m")),
            "track_width_front": float(self.value("geometry.front_track_m")),
            "track_width_rear": float(self.value("geometry.rear_track_m")),
            "max_steer": float(self.value("actuators.steering.max_angle_rad")),
            "max_steer_rate": float(self.value("actuators.steering.max_rate_rad_s")),
            "max_speed": float(
                self.value("powertrain.longitudinal_limits.max_speed_mps")
            ),
            "max_throttle": float(self.value("actuators.throttle.max_value")),
            "max_brake": float(self.value("actuators.brake.max_value")),
            "max_accel": float(
                self.value("powertrain.longitudinal_limits.max_accel_low_speed_mps2")
            ),
            "max_decel": float(self.value("brakes.max_decel_mps2")),
            "steering_time_constant": float(
                self.value("actuators.steering.first_order_response_time_s")
            ),
            "throttle_time_constant": float(
                self.value("actuators.throttle.first_order_response_time_s")
            ),
            "brake_time_constant": float(
                self.value("actuators.brake.first_order_response_time_s")
            ),
        }

    def copy_data(self) -> dict[str, Any]:
        return deepcopy(self.data)


def expand_a2rl_component_config(config: dict[str, Any]) -> dict[str, Any]:
    """Add legacy controller values while retaining the model-root selector."""

    if "model_root" not in config:
        return config
    loaded = A2RLVehicleConfig.from_component_config(config)
    expanded = deepcopy(config)
    expanded.update(loaded.legacy_flat_config())
    return expanded


def _validate_sources(node: Any, path: str = "") -> None:
    if isinstance(node, dict):
        if "value" in node:
            source = node.get("source")
            if source not in ALLOWED_SOURCES:
                label = path or "<root>"
                raise ValueError(
                    f"{label} must use one of {sorted(ALLOWED_SOURCES)}, got {source!r}"
                )
        for key, value in node.items():
            _validate_sources(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _validate_sources(value, f"{path}[{index}]")

