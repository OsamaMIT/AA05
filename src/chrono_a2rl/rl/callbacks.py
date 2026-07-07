"""Stable-Baselines3 callback helpers."""

from __future__ import annotations


def stable_baselines3_available() -> bool:
    """Return whether SB3 can be imported."""

    try:
        import stable_baselines3  # noqa: F401
    except ImportError:
        return False
    return True
