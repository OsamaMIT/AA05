from __future__ import annotations

from chrono_a2rl.common.config import load_experiment_config, load_yaml


def test_vehicle_config_uses_public_a2rl_profile() -> None:
    config = load_experiment_config("configs/experiments/mpc_yas_marina_flat.yaml")
    vehicle = config["vehicle"]
    assert vehicle["name"] == "a2rl_eav25_style"
    assert vehicle["public_reference"]["model_family"] == "Dallara EAV24 / EAV25"
    assert vehicle["public_reference"]["base_platform"] == "Dallara Super Formula car"
    assert vehicle["wheelbase"] == 3.115


def test_sensor_profile_documents_public_a2rl_stack() -> None:
    sensors = load_yaml("vehicles/a2rl_style_sf23/sensor_params.yaml")
    assert sensors["public_sensor_suite"]["cameras"]["count"] == 7
    assert sensors["public_sensor_suite"]["radars"]["count"] == 4
    assert sensors["public_sensor_suite"]["lidars"]["count"] == 3
