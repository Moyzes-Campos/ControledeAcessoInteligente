from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


Point = tuple[float, float]
DEFAULT_LINE_NORMALIZED = (0.489844, 0.155556, 0.500391, 0.890972)


@dataclass(frozen=True, slots=True)
class LineSettings:
    normalized: tuple[float, float, float, float] = DEFAULT_LINE_NORMALIZED
    invert: bool = False
    hysteresis: float = 18.0

    def __post_init__(self):
        if len(self.normalized) != 4 or not all(
            math.isfinite(value) and 0 <= value <= 1 for value in self.normalized
        ):
            raise ValueError("A linha deve conter quatro coordenadas entre 0 e 1")
        if self.normalized[:2] == self.normalized[2:]:
            raise ValueError("A linha precisa de dois pontos diferentes")
        if not math.isfinite(self.hysteresis) or self.hysteresis < 0:
            raise ValueError("A histerese da linha deve ser finita e nao negativa")

    @classmethod
    def load(cls, path: Path | None, invert: bool = False):
        if path is None or not path.exists():
            return cls(invert=invert)
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return cls(tuple(data["line_normalized"]),
                   bool(data.get("invert", False)) ^ invert,
                   float(data.get("hysteresis_pixels", 18.0)))

    def pixels(self, width: int, height: int) -> tuple[Point, Point]:
        x1, y1, x2, y2 = self.normalized
        return (x1 * width, y1 * height), (x2 * width, y2 * height)


def signed_distance(point: Point, start: Point, end: Point) -> float:
    """Signed perpendicular distance; the sign identifies the line side."""
    px, py = point
    x1, y1 = start
    x2, y2 = end
    length = max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1e-9)
    return ((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)) / length


def projection_fraction(point: Point, start: Point, end: Point) -> float:
    px, py = point
    x1, y1 = start
    dx, dy = end[0] - x1, end[1] - y1
    length_squared = max(dx * dx + dy * dy, 1e-9)
    return ((px - x1) * dx + (py - y1) * dy) / length_squared


@dataclass(slots=True)
class Crossing:
    track_id: int
    direction: str
    point: Point


class LineCounter:
    def __init__(self, start: Point, end: Point, hysteresis: float = 18.0, invert: bool = False):
        self.start = start
        self.end = end
        self.hysteresis = max(0.0, hysteresis)
        self.invert = invert
        self._stable_side: dict[int, int] = {}
        self._stable_point: dict[int, Point] = {}
        self.entries = 0
        self.exits = 0

    def update(self, track_id: int, point: Point) -> Crossing | None:
        distance = signed_distance(point, self.start, self.end)
        if distance == 0 or abs(distance) < self.hysteresis:
            return None
        side = 1 if distance > 0 else -1
        previous = self._stable_side.get(track_id)
        previous_point = self._stable_point.get(track_id)
        self._stable_side[track_id] = side
        self._stable_point[track_id] = point
        if previous is None or previous == side:
            return None
        previous_distance = signed_distance(previous_point, self.start, self.end)
        fraction = previous_distance / (previous_distance - distance)
        intersection = (previous_point[0] + fraction * (point[0] - previous_point[0]),
                        previous_point[1] + fraction * (point[1] - previous_point[1]))
        if not 0.0 <= projection_fraction(intersection, self.start, self.end) <= 1.0:
            return None
        direction = "entrada" if previous < side else "saida"
        if self.invert:
            direction = "saida" if direction == "entrada" else "entrada"
        if direction == "entrada":
            self.entries += 1
        else:
            self.exits += 1
        return Crossing(track_id, direction, intersection)

    def prune(self, active_track_ids: set[int]) -> None:
        for track_id in list(self._stable_side):
            if track_id not in active_track_ids:
                del self._stable_side[track_id]
                del self._stable_point[track_id]

