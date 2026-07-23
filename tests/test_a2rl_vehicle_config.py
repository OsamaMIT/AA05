from __future__ import annotations

import math

from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.vehicle.a2rl_vehicle_config import (
    ALLOWED_SOURCES,
    A2RLVehicleConfig,
)


def test_a2rl_vehicle_config_loads_public_and_proxy_values() -> None:
    config = A2RLVehicleConfig.load()

    assert config.value("vehicle.mass.total_kg") == 690.0
    assert config.source("vehicle.mass.total_kg") == "public"
    assert config.source("geometry.wheelbase_m") == "proxy_sf19_sf23"
    assert math.isclose(
        config.value("geometry.cg_to_front_axle_m")
        + config.value("geometry.cg_to_rear_axle_m"),
        config.value("geometry.wheelbase_m"),
    )


def test_all_value_records_use_allowed_provenance() -> None:
    config = A2RLVehicleConfig.load()

    def visit(node) -> None:
        if isinstance(node, dict):
            if "value" in node:
                assert node["source"] in ALLOWED_SOURCES
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(config.data)


def test_dynamic_experiment_expands_legacy_controller_values() -> None:
    config = load_experiment_config(
        "configs/experiments/a2rl_dynamic_vehicle_yas_marina.yaml"
    )

    assert config["vehicle"]["mass"] == 690.0
    assert config["vehicle"]["wheelbase"] == 3.115
    assert config["vehicle"]["dynamics_level"] == 2
    assert config["simulation"]["chrono_mode"] == "dynamic_bicycle"

