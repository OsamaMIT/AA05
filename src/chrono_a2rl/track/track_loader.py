"""Load TUMFTM-style track CSV files or create synthetic fallback tracks."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from chrono_a2rl.common.config import resolve_path
from chrono_a2rl.common.logging import get_logger
from chrono_a2rl.track.curbs import load_curbs
from chrono_a2rl.track.track_geometry import TrackGeometry


LOGGER = get_logger(__name__)


_CENTER_X = ["x_m", "x", "center_x", "x_center", "centerline_x", "refline_x", "# x_m"]
_CENTER_Y = ["y_m", "y", "center_y", "y_center", "centerline_y", "refline_y"]
_RACELINE_X = ["x_raceline_m", "x_race_m", "raceline_x", "x_opt_m", "rx"]
_RACELINE_Y = ["y_raceline_m", "y_race_m", "raceline_y", "y_opt_m", "ry"]
_WIDTH_LEFT = ["w_tr_left_m", "width_left", "left_width", "wl", "track_width_left"]
_WIDTH_RIGHT = ["w_tr_right_m", "width_right", "right_width", "wr", "track_width_right"]


def load_track_from_config(config: dict[str, Any]) -> TrackGeometry:
    """Load a track from config, using a synthetic fallback when needed."""

    csv_value = config.get("csv_path")
    raceline_value = config.get("raceline_path")
    curbs_path = config.get("curbs_path")
    curbs = load_curbs(resolve_path(curbs_path) if curbs_path else None)
    if csv_value:
        csv_path = resolve_path(csv_value)
        if csv_path.exists():
            raceline_path = resolve_path(raceline_value) if raceline_value else None
            return load_tumftm_csv(
                csv_path,
                name=str(config.get("name", csv_path.stem)),
                use_raceline_if_available=bool(config.get("use_raceline_if_available", True)),
                raceline_path=raceline_path if raceline_path and raceline_path.exists() else None,
                curbs=curbs,
            )

    fallback = config.get("fallback", {})
    if fallback.get("enabled", True):
        LOGGER.warning(
            "Track CSV is missing. Using synthetic fallback. To use Yas Marina, "
            "process TUMFTM racetrack-database data into %s.",
            csv_value,
        )
        return create_synthetic_track(fallback, curbs=curbs)

    raise FileNotFoundError(
        f"Track CSV not found: {csv_value}. Provide TUMFTM data or enable fallback."
    )


def load_tumftm_csv(
    path: str | Path,
    *,
    name: str,
    use_raceline_if_available: bool = True,
    raceline_path: str | Path | None = None,
    curbs=None,
) -> TrackGeometry:
    """Load a flexible TUMFTM-style CSV.

    The loader infers common column names for centerline, optional raceline,
    and left/right widths. Whitespace, comma, and semicolon separated files are
    supported through pandas' Python parser.
    """

    csv_path = Path(path)
    df = _read_csv_flexible(csv_path)
    df = df.rename(columns={str(col).strip(): str(col).strip() for col in df.columns})
    df = _coerce_numeric(df)
    if len(df.columns) == 1:
        df = pd.read_csv(csv_path, comment="#", sep=r"\s+", engine="python")
        df = df.rename(columns={str(col).strip(): str(col).strip() for col in df.columns})
        df = _coerce_numeric(df)

    x_col = _find_column(df, _CENTER_X)
    y_col = _find_column(df, _CENTER_Y)
    if (x_col is None or y_col is None) and _columns_look_numeric(df):
        df = pd.read_csv(csv_path, comment="#", sep=None, engine="python", header=None)
        if len(df.columns) == 1:
            df = pd.read_csv(csv_path, comment="#", sep=r"\s+", engine="python", header=None)
        df = df.rename(columns={col: f"col_{idx}" for idx, col in enumerate(df.columns)})
        df = _coerce_numeric(df)
        x_col = _find_column(df, _CENTER_X)
        y_col = _find_column(df, _CENTER_Y)
    if x_col is None or y_col is None:
        numeric = df.select_dtypes(include=["number"])
        if numeric.shape[1] < 2:
            raise ValueError(f"Could not infer centerline columns from {csv_path}")
        x_col, y_col = numeric.columns[:2]

    left_col = _find_column(df, _WIDTH_LEFT)
    right_col = _find_column(df, _WIDTH_RIGHT)
    width_left = df[left_col].to_numpy(float) if left_col else 6.0
    width_right = df[right_col].to_numpy(float) if right_col else 6.0

    centerline = df[[x_col, y_col]].to_numpy(float)
    raceline = None
    if use_raceline_if_available:
        rx = _find_column(df, _RACELINE_X)
        ry = _find_column(df, _RACELINE_Y)
        if rx is not None and ry is not None:
            raceline = df[[rx, ry]].to_numpy(float)
        elif raceline_path is not None:
            raceline = _load_raceline_csv(Path(raceline_path))

    return TrackGeometry(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        name=name,
        raceline=raceline,
        curbs=curbs,
    )


def create_synthetic_track(config: dict[str, Any] | None = None, *, curbs=None) -> TrackGeometry:
    """Create a smooth closed-loop oval used for tests and offline development."""

    cfg = config or {}
    n = int(cfg.get("num_points", 400))
    radius_x = float(cfg.get("radius_x", 80.0))
    radius_y = float(cfg.get("radius_y", 52.0))
    width_left = float(cfg.get("width_left", 8.0))
    width_right = float(cfg.get("width_right", 8.0))
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    centerline = np.column_stack([radius_x * np.cos(theta), radius_y * np.sin(theta)])
    return TrackGeometry(
        centerline=centerline,
        width_left=width_left,
        width_right=width_right,
        name=str(cfg.get("name", "synthetic_oval")),
        curbs=curbs,
    )


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return str(normalized[key])
    return None


def _load_raceline_csv(path: Path) -> np.ndarray:
    df = _read_csv_flexible(path)
    df = df.rename(columns={str(col).strip(): str(col).strip() for col in df.columns})
    df = _coerce_numeric(df)
    x_col = _find_column(df, _RACELINE_X + _CENTER_X)
    y_col = _find_column(df, _RACELINE_Y + _CENTER_Y)
    if x_col is None or y_col is None:
        numeric = df.select_dtypes(include=["number"])
        if numeric.shape[1] < 2:
            raise ValueError(f"Could not infer raceline columns from {path}")
        x_col, y_col = numeric.columns[:2]
    return df[[x_col, y_col]].to_numpy(float)


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    first_line = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                first_line = line.strip()
                break
    if first_line.startswith("#") and any(name in first_line.lower() for name in ("x_m", "y_m")):
        header = first_line.lstrip("#").strip()
        sep_pattern = _delimiter_pattern(header)
        names = [part.strip() for part in re.split(sep_pattern, header) if part.strip()]
        return pd.read_csv(
            path,
            comment="#",
            names=names,
            sep=sep_pattern,
            engine="python",
        )
    return pd.read_csv(path, comment="#", sep=None, engine="python")


def _delimiter_pattern(line: str) -> str:
    if ";" in line:
        return ";"
    if "," in line:
        return ","
    return r"\s+"


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        converted = pd.to_numeric(result[column], errors="coerce")
        if int(converted.notna().sum()) == int(result[column].notna().sum()):
            result[column] = converted
    return result.dropna(how="all")


def _columns_look_numeric(df: pd.DataFrame) -> bool:
    numeric_columns = pd.to_numeric(pd.Index(df.columns), errors="coerce")
    return int(numeric_columns.notna().sum()) >= 2
