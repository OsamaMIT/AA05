"""Lightweight replay helpers for saved CSV logs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_replay(path: str | Path) -> pd.DataFrame:
    """Load a rollout CSV for analysis or plotting."""

    return pd.read_csv(path)
