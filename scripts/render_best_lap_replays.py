#!/usr/bin/env python3
"""Render compact full-track and follow-camera GIFs from a completed lap log."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chrono_a2rl.common.config import load_experiment_config
from chrono_a2rl.evaluation.metrics import format_lap_time
from chrono_a2rl.track.track_geometry import TrackGeometry
from chrono_a2rl.track.track_loader import load_track_from_config


CANVAS = (720, 480)
MAP_BOX = (18, 48, 540, 462)
PANEL_X = 558
COLORS = {
    "background": "#f4f5f3",
    "panel": "#ffffff",
    "limit": "#202425",
    "center": "#afb3b1",
    "raceline": "#22a447",
    "trail": "#2474b5",
    "car": "#df242b",
    "text": "#171a1b",
    "muted": "#62696a",
}


def render_replays(
    *,
    log_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    fps: int = 10,
    replay_speed: float = 4.0,
    follow_radius: float = 120.0,
    follow_ahead: float = 35.0,
) -> dict[str, str]:
    """Render both README camera views from one completed evaluation CSV."""

    config = load_experiment_config(config_path)
    track = load_track_from_config(config["track"])
    data = pd.read_csv(log_path)
    _validate_log(data)
    rows = _sample_rows(data, fps=fps, replay_speed=replay_speed)
    destination = (ROOT / output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    outputs = {
        camera: destination / f"best_lap_{camera}.gif"
        for camera in ("full", "follow")
    }
    for camera, output_path in outputs.items():
        _render_camera(
            rows,
            track,
            output_path,
            camera=camera,
            fps=fps,
            follow_radius=follow_radius,
            follow_ahead=follow_ahead,
        )
    return {camera: str(path) for camera, path in outputs.items()}


def _validate_log(data: pd.DataFrame) -> None:
    required = {
        "sim_time_seconds",
        "x",
        "y",
        "yaw",
        "speed_kmh",
        "target_speed_kmh",
        "throttle",
        "brake",
        "s",
        "progress_s",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"evaluation log is missing columns: {', '.join(missing)}")
    if (
        "crossed_start_finish" in data
        and not data["crossed_start_finish"].astype(bool).any()
    ):
        raise ValueError("evaluation log does not contain a start/finish crossing")
    if "termination_reason" in data:
        reasons = set(data["termination_reason"].dropna().astype(str))
        if reasons and "lap_completed" not in reasons:
            raise ValueError("evaluation log is not a completed lap")


def _sample_rows(
    data: pd.DataFrame,
    *,
    fps: int,
    replay_speed: float,
) -> pd.DataFrame:
    if fps <= 0 or replay_speed <= 0.0:
        raise ValueError("fps and replay_speed must be positive")
    times = data["sim_time_seconds"].to_numpy(float)
    targets = np.arange(times[0], times[-1], replay_speed / fps)
    indices = np.searchsorted(times, targets, side="left")
    indices = np.unique(
        np.clip(np.append(indices, len(data) - 1), 0, len(data) - 1)
    )
    return data.iloc[indices].reset_index(drop=True)


def _render_camera(
    data: pd.DataFrame,
    track: TrackGeometry,
    output_path: Path,
    *,
    camera: str,
    fps: int,
    follow_radius: float,
    follow_ahead: float,
) -> None:
    geometry = _track_geometry(track)
    full_bounds = _bounds(
        np.concatenate([geometry["left_x"], geometry["right_x"]]),
        np.concatenate([geometry["left_y"], geometry["right_y"]]),
        padding=35.0,
    )
    total_progress = max(float(data["progress_s"].iloc[-1]), track.length)
    frames: list[Image.Image] = []

    for index, row in data.iterrows():
        bounds = full_bounds
        if camera == "follow":
            center_x = float(row["x"]) + follow_ahead * np.cos(float(row["yaw"]))
            center_y = float(row["y"]) + follow_ahead * np.sin(float(row["yaw"]))
            bounds = (
                center_x - follow_radius,
                center_x + follow_radius,
                center_y - follow_radius,
                center_y + follow_radius,
            )
        frame = _render_frame(
            row,
            data.iloc[: index + 1],
            track,
            geometry,
            bounds,
            camera,
            total_progress,
        )
        frames.append(
            frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=96)
        )

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )


def _render_frame(
    row: pd.Series,
    history: pd.DataFrame,
    track: TrackGeometry,
    geometry: dict[str, np.ndarray],
    bounds: tuple[float, float, float, float],
    camera: str,
    total_progress: float,
) -> Image.Image:
    image = Image.new("RGB", CANVAS, COLORS["background"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, CANVAS[0], 38), fill="#171a1b")
    draw.text(
        (18, 8),
        "A2RL Chrono | Yas Marina best lap",
        fill="#ffffff",
        font=_font(20, bold=True),
    )

    map_image = Image.new("RGB", CANVAS, COLORS["background"])
    map_draw = ImageDraw.Draw(map_image)
    transform = _transform(bounds, MAP_BOX)
    _line(map_draw, geometry["left_x"], geometry["left_y"], transform, "limit", 3)
    _line(map_draw, geometry["right_x"], geometry["right_y"], transform, "limit", 3)
    _line(map_draw, geometry["center_x"], geometry["center_y"], transform, "center", 1)
    if track.raceline is not None:
        _line(
            map_draw,
            track.raceline[:, 0],
            track.raceline[:, 1],
            transform,
            "raceline",
            2,
        )
    _draw_start_line(map_draw, track, transform)

    trail = history.tail(180 if camera == "follow" else len(history))
    _line(
        map_draw,
        trail["x"].to_numpy(float),
        trail["y"].to_numpy(float),
        transform,
        "trail",
        3,
        closed=False,
    )
    car_x, car_y = transform(float(row["x"]), float(row["y"]))
    heading_length = 16.0 if camera == "follow" else 28.0
    nose = transform(
        float(row["x"]) + heading_length * np.cos(float(row["yaw"])),
        float(row["y"]) + heading_length * np.sin(float(row["yaw"])),
    )
    map_draw.line((car_x, car_y, *nose), fill=COLORS["car"], width=4)
    map_draw.ellipse(
        (car_x - 6, car_y - 6, car_x + 6, car_y + 6),
        fill=COLORS["car"],
        outline="#ffffff",
        width=2,
    )
    image.paste(map_image.crop(MAP_BOX), (MAP_BOX[0], MAP_BOX[1]))
    draw.rectangle(
        (PANEL_X - 8, 38, CANVAS[0], CANVAS[1]),
        fill=COLORS["panel"],
    )
    _draw_panel(draw, row, camera, total_progress)
    return image


def _draw_start_line(draw: ImageDraw.ImageDraw, track, transform) -> None:
    start = track.interpolate(0.0)
    normal = np.array([-np.sin(start.heading), np.cos(start.heading)])
    left = np.array([start.x, start.y]) + normal * start.width_left
    right = np.array([start.x, start.y]) - normal * start.width_right
    draw.line((*transform(*left), *transform(*right)), fill="#000000", width=5)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    row: pd.Series,
    camera: str,
    total_progress: float,
) -> None:
    draw.text(
        (PANEL_X + 8, 55),
        f"{camera.upper()} CAMERA",
        fill=COLORS["muted"],
        font=_font(11),
    )
    fields = [
        ("LAP TIME", format_lap_time(float(row["sim_time_seconds"]))),
        ("SPEED", f"{float(row['speed_kmh']):.1f} km/h"),
        ("PROFILE", f"{float(row['target_speed_kmh']):.1f} km/h"),
        ("THROTTLE", f"{float(row['throttle']):.2f}"),
        ("BRAKE", f"{float(row['brake']):.2f}"),
        ("PROGRESS", f"{100.0 * float(row['progress_s']) / total_progress:.1f}%"),
    ]
    y = 86
    for label, value in fields:
        draw.text(
            (PANEL_X + 8, y),
            label,
            fill=COLORS["muted"],
            font=_font(11),
        )
        draw.text(
            (PANEL_X + 8, y + 15),
            value,
            fill=COLORS["text"],
            font=_font(14, bold=True),
        )
        y += 53
    draw.text(
        (PANEL_X + 8, 423),
        "4x replay",
        fill=COLORS["muted"],
        font=_font(14),
    )
    draw.text(
        (PANEL_X + 8, 444),
        "Chrono backend",
        fill=COLORS["muted"],
        font=_font(11),
    )


def _track_geometry(track: TrackGeometry) -> dict[str, np.ndarray]:
    arrays = track.sample_arrays()
    x, y = arrays["x"], arrays["y"]
    normal_x = -np.sin(arrays["heading"])
    normal_y = np.cos(arrays["heading"])
    return {
        "center_x": x,
        "center_y": y,
        "left_x": x + normal_x * arrays["width_left"],
        "left_y": y + normal_y * arrays["width_left"],
        "right_x": x - normal_x * arrays["width_right"],
        "right_y": y - normal_y * arrays["width_right"],
    }


def _bounds(x, y, *, padding: float) -> tuple[float, float, float, float]:
    return (
        float(np.min(x) - padding),
        float(np.max(x) + padding),
        float(np.min(y) - padding),
        float(np.max(y) + padding),
    )


def _transform(bounds, box):
    min_x, max_x, min_y, max_y = bounds
    left, top, right, bottom = box
    scale = min(
        (right - left) / max(max_x - min_x, 1.0),
        (bottom - top) / max(max_y - min_y, 1.0),
    )
    center_x, center_y = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)
    pixel_x, pixel_y = 0.5 * (left + right), 0.5 * (top + bottom)

    def convert(x: float, y: float) -> tuple[int, int]:
        return (
            int(round(pixel_x + (x - center_x) * scale)),
            int(round(pixel_y - (y - center_y) * scale)),
        )

    return convert


def _line(
    draw: ImageDraw.ImageDraw,
    x,
    y,
    transform,
    color: str,
    width: int,
    *,
    closed: bool = True,
) -> None:
    points = [transform(float(px), float(py)) for px, py in zip(x, y)]
    if len(points) < 2:
        return
    if closed:
        points.append(points[0])
    draw.line(points, fill=COLORS.get(color, color), width=width, joint="curve")


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        required=True,
        help="Completed `aa eval` CSV used as the replay source.",
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/rl_planner_yas_marina.yaml",
    )
    parser.add_argument("--output-dir", default="docs/assets/replays")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--replay-speed", type=float, default=4.0)
    parser.add_argument("--follow-radius", type=float, default=120.0)
    parser.add_argument("--follow-ahead", type=float, default=35.0)
    args = parser.parse_args()
    outputs = render_replays(
        log_path=args.log,
        config_path=args.config,
        output_dir=args.output_dir,
        fps=args.fps,
        replay_speed=args.replay_speed,
        follow_radius=args.follow_radius,
        follow_ahead=args.follow_ahead,
    )
    for camera, path in outputs.items():
        print(f"{camera}: {path}")


if __name__ == "__main__":
    main()
