"""Version-tolerant PyChrono factory for the approximate EAV24 rigid body."""

from __future__ import annotations

from chrono_a2rl.chrono_interface.a2rl_chrono_body import A2RLChronoBody
from chrono_a2rl.vehicle.a2rl_vehicle_config import A2RLVehicleConfig


def create_a2rl_chrono_body(chrono, config: A2RLVehicleConfig):
    system_cls = getattr(chrono, "ChSystemNSC", None) or getattr(
        chrono, "ChSystem", None
    )
    if system_cls is None:
        raise RuntimeError("PyChrono does not expose a supported ChSystem")
    system = system_cls()
    vec_cls = getattr(chrono, "ChVector3d", None) or getattr(
        chrono, "ChVectorD", None
    )
    gravity = vec_cls(0.0, 0.0, -9.81)
    if hasattr(system, "SetGravitationalAcceleration"):
        system.SetGravitationalAcceleration(gravity)
    elif hasattr(system, "Set_G_acc"):
        system.Set_G_acc(gravity)

    body = chrono.ChBody()
    body.SetMass(float(config.value("vehicle.mass.total_kg")))
    body.SetInertiaXX(
        vec_cls(
            float(config.value("vehicle.inertia.ix_kgm2")),
            float(config.value("vehicle.inertia.iy_kgm2")),
            float(config.value("vehicle.inertia.iz_kgm2")),
        )
    )
    _set_fixed(body, False)
    _add_visual_box(
        chrono,
        body,
        float(config.value("geometry.length_m")) * 0.62,
        float(config.value("geometry.width_m")) * 0.58,
        0.45,
    )
    _add_body(system, body)

    ground = chrono.ChBody()
    ground.SetMass(1.0)
    _set_fixed(ground, True)
    ground.SetPos(vec_cls(0.0, 0.0, -0.08))
    _add_body(system, ground)
    return system, A2RLChronoBody(chrono, system, body)


def _add_visual_box(chrono, body, length: float, width: float, height: float) -> None:
    shape_cls = getattr(chrono, "ChVisualShapeBox", None)
    if shape_cls is None or not hasattr(body, "AddVisualShape"):
        return
    try:
        shape = shape_cls(length, width, height)
    except TypeError:
        shape = shape_cls()
        if hasattr(shape, "SetSize"):
            vec_cls = getattr(chrono, "ChVector3d", None) or getattr(
                chrono, "ChVectorD", None
            )
            shape.SetSize(vec_cls(length, width, height))
    body.AddVisualShape(shape)


def _set_fixed(body, fixed: bool) -> None:
    if hasattr(body, "SetFixed"):
        body.SetFixed(fixed)
    else:
        body.SetBodyFixed(fixed)


def _add_body(system, body) -> None:
    if hasattr(system, "AddBody"):
        system.AddBody(body)
    else:
        system.Add(body)
