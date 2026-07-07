"""Small numeric helpers shared across modules."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a scalar value."""

    return float(max(lower, min(upper, value)))


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def circular_moving_average(values: Iterable[float], window: int) -> np.ndarray:
    """Smooth a closed-loop signal with a centered circular moving average."""

    arr = np.asarray(list(values), dtype=float)
    if window <= 1 or arr.size == 0:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.concatenate([arr[-radius:], arr, arr[:radius]])
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def has_nan(*values: float) -> bool:
    """Return true if any scalar is NaN or infinite."""

    return any(not math.isfinite(float(v)) for v in values)
