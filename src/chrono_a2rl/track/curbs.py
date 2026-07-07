"""Optional semantic curb representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class Curb:
    """Semantic curb interval."""

    side: str
    s_start: float
    s_end: float
    width: float
    height: float
    friction: float
    type: str
    penalty_weight: float
    legal_status: str


@dataclass(slots=True)
class CurbContact:
    """Semantic curb contact result."""

    on_curb: bool = False
    side: str = ""
    penalty_weight: float = 0.0
    curb_type: str = ""


class CurbMap:
    """Collection of semantic curb intervals."""

    def __init__(self, curbs: list[Curb], curb_level: int = 0) -> None:
        self.curbs = curbs
        self.curb_level = curb_level

    def is_on_curb(
        self,
        s: float,
        n: float,
        width_left: float | None = None,
        width_right: float | None = None,
    ) -> bool:
        """Return whether a Frenet point is on a semantic curb."""

        return self.contact(s, n, width_left, width_right).on_curb

    def contact(
        self,
        s: float,
        n: float,
        width_left: float | None = None,
        width_right: float | None = None,
    ) -> CurbContact:
        """Return semantic curb contact and penalty metadata.

        For level-1 curbs, flat curb zones live near the track boundaries. The
        left curb occupies `[width_left - curb.width, width_left]`; the right
        curb occupies `[-width_right, -width_right + curb.width]`.
        """

        if self.curb_level <= 0:
            return CurbContact()
        for curb in self.curbs:
            if not _s_in_interval(s, curb.s_start, curb.s_end):
                continue
            side = curb.side.lower()
            if side == "left" and _on_left_curb(n, curb.width, width_left):
                return CurbContact(
                    on_curb=True,
                    side="left",
                    penalty_weight=curb.penalty_weight,
                    curb_type=curb.type,
                )
            if side == "right" and _on_right_curb(n, curb.width, width_right):
                return CurbContact(
                    on_curb=True,
                    side="right",
                    penalty_weight=curb.penalty_weight,
                    curb_type=curb.type,
                )
        return CurbContact()


def _s_in_interval(s: float, start: float, end: float) -> bool:
    if start == end:
        return False
    if start < end:
        return start <= s <= end
    return s >= start or s <= end


def _on_left_curb(n: float, curb_width: float, width_left: float | None) -> bool:
    if curb_width <= 0.0:
        return False
    if width_left is None:
        return 0.0 < n <= curb_width
    return max(0.0, width_left - curb_width) <= n <= width_left


def _on_right_curb(n: float, curb_width: float, width_right: float | None) -> bool:
    if curb_width <= 0.0:
        return False
    if width_right is None:
        return -curb_width <= n < 0.0
    return -width_right <= n <= min(0.0, -width_right + curb_width)


def load_curbs(path: str | Path | None) -> CurbMap:
    """Load optional curbs YAML."""

    if path is None:
        return CurbMap([], curb_level=0)
    curb_path = Path(path)
    if not curb_path.exists():
        return CurbMap([], curb_level=0)
    with curb_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    curbs = [
        Curb(
            side=str(item.get("side", "left")),
            s_start=float(item.get("s_start", 0.0)),
            s_end=float(item.get("s_end", 0.0)),
            width=float(item.get("width", 0.0)),
            height=float(item.get("height", 0.0)),
            friction=float(item.get("friction", 1.0)),
            type=str(item.get("type", "semantic")),
            penalty_weight=float(item.get("penalty_weight", 0.0)),
            legal_status=str(item.get("legal_status", "unknown")),
        )
        for item in data.get("curbs", [])
    ]
    return CurbMap(curbs, curb_level=int(data.get("curb_level", 0)))
