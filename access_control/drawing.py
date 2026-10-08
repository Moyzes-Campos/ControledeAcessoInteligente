from __future__ import annotations

import cv2
import numpy as np
import time

from .access import AccessState
from .line_crossing import LineCounter
from .tracking import bbox_center

from .types import FaceResult, PoseDetection, TrackIdentity, TrackedPose


SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2),
    (1, 3), (2, 4),
)


def draw_scene(
    frame: np.ndarray,
    detections: list[PoseDetection],
    tracks: list[TrackedPose],
    identities: dict[int, TrackIdentity],
    faces: list[FaceResult],
    fps: float,
    processing_ms: float | None = None,
    coordinate_scale: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    canvas = frame.copy()
    sx, sy = coordinate_scale
    for track in tracks:
        detection = detections[track.pose_index]
        identity = identities[track.track_id]
        color = _track_color(track.track_id)
        _draw_pose(canvas, detection, color, coordinate_scale)
        x1, y1, x2, y2 = (int(round(value * scale)) for value, scale in
                          zip(track.bbox_xyxy, (sx, sy, sx, sy)))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        name = identity.label if identity.known else "Desconhecido"
        text = f"ID {track.track_id} | {name} | pessoa {track.confidence:.2f}"
        _label(canvas, text, (x1, max(24, y1)), color)

    for face in faces:
        x, y, w, h = (round(value * scale) for value, scale in
                       zip(face.bbox_xywh, (sx, sy, sx, sy)))
        color = (0, 255, 0)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        score = f"{face.score:.2f}" if face.score >= 0 else "sem cadastro"
        _label(canvas, f"{face.label} {score}", (x, max(22, y)), color, scale=0.52)

    status = f"Pessoas: {len(tracks)} | FPS: {fps:.1f}"
    if processing_ms is not None:
        status += f" | Processamento: {processing_ms:.0f} ms"
    _label(canvas, status, (18, 32), (30, 30, 30), scale=0.65)
    return canvas


def draw_access_overlay(canvas: np.ndarray, tracks: list[TrackedPose], counter: LineCounter,
                        state: AccessState, esp_status: dict | None,
                        coordinate_scale: tuple[float, float] = (1.0, 1.0)):
    color = (0, 255, 255)
    sx, sy = coordinate_scale
    start = (round(counter.start[0] * sx), round(counter.start[1] * sy))
    end = (round(counter.end[0] * sx), round(counter.end[1] * sy))
    cv2.line(canvas, start, end, color, 3, cv2.LINE_AA)
    for point in (start, end):
        cv2.circle(canvas, point, 7, color, -1, cv2.LINE_AA)
    # Arrow points to the positive side (entry), or the negative side when inverted.
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max((dx * dx + dy * dy) ** .5, 1)
    sign = -1 if counter.invert else 1
    middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    arrow = (round(middle[0] - sign * dy / length * 80),
             round(middle[1] + sign * dx / length * 80))
    cv2.arrowedLine(canvas, tuple(round(v) for v in middle), arrow, color, 3, cv2.LINE_AA)
    _label(canvas, "ENTRADA", arrow, (70, 70, 0), scale=.55)
    for track in tracks:
        x, y = bbox_center(track.bbox_xyxy)
        cv2.circle(canvas, (round(x * sx), round(y * sy)), 5, color, -1)
    _label(canvas, f"Entradas: {state.entries} | Saidas: {state.exits} | "
           f"Visitas abertas: {state.open_visits}", (18, 64), (30, 30, 30), scale=.65)
    lamps = {"orange": (0, 150, 255), "red": (0, 0, 220), "green": (0, 150, 0)}
    names = {"orange": "LARANJA", "red": "VERMELHO", "green": "VERDE"}
    signal = names[state.lamp]
    if state.alarm_until > time.monotonic():
        signal += " | ALERTA SONORO"
    esp = ("desativado" if esp_status is None else
           "conectado" if esp_status["connected"] else "desconectado")
    _label(canvas, f"Sinal: {signal} | ESP32: {esp}", (18, 96), lamps[state.lamp], scale=.6)


def _draw_pose(frame: np.ndarray, detection: PoseDetection, color: tuple[int, int, int],
               coordinate_scale: tuple[float, float] = (1.0, 1.0)) -> None:
    points = detection.keypoints_xyc
    if len(points) < 17:
        return
    scale = np.asarray(coordinate_scale)
    for first, second in SKELETON:
        if points[first, 2] >= 0.35 and points[second, 2] >= 0.35:
            p1 = tuple(np.rint(points[first, :2] * scale).astype(int))
            p2 = tuple(np.rint(points[second, :2] * scale).astype(int))
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    for x, y, confidence in points:
        if confidence >= 0.35:
            cv2.circle(frame, (round(x * scale[0]), round(y * scale[1])), 3,
                       (245, 245, 245), -1, cv2.LINE_AA)


def _label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.58,
) -> None:
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, baseline_y = origin
    top = max(0, baseline_y - height - 8)
    cv2.rectangle(frame, (x, top), (x + width + 10, baseline_y + baseline), color, -1)
    cv2.putText(frame, text, (x + 5, baseline_y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _track_color(track_id: int) -> tuple[int, int, int]:
    return ((37 * track_id + 60) % 205 + 30, (83 * track_id + 40) % 205 + 30, (131 * track_id + 20) % 205 + 30)

