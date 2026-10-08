from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import PoseDetection, TrackedPose


@dataclass(slots=True)
class _TrackState:
    track_id: int
    bbox_xyxy: np.ndarray
    velocity_xy: np.ndarray
    last_center_xy: tuple[float, float]
    hits: int = 1
    age: int = 1
    missed_frames: int = 0


class OCSortPersonTracker:
    """OC-SORT enxuto: IoU, predicao de movimento e consistencia de direcao."""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_age: int = 20,
        velocity_weight: float = 0.20,
        velocity_smoothing: float = 0.65,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_age = max(1, int(max_age))
        self.velocity_weight = max(0.0, float(velocity_weight))
        self.velocity_smoothing = min(0.95, max(0.0, float(velocity_smoothing)))
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    @property
    def active_track_ids(self) -> set[int]:
        return set(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(self, detections: list[PoseDetection]) -> list[TrackedPose]:
        predicted = {track_id: self._predict(track) for track_id, track in self._tracks.items()}
        candidates: list[tuple[float, float, int, int]] = []

        for track_id, predicted_bbox in predicted.items():
            track = self._tracks[track_id]
            for detection_index, detection in enumerate(detections):
                iou = bbox_iou(predicted_bbox, detection.bbox_xyxy)
                if iou < self.iou_threshold:
                    continue
                direction = _direction_consistency(
                    track.velocity_xy,
                    track.last_center_xy,
                    bbox_center(detection.bbox_xyxy),
                )
                candidates.append((iou + self.velocity_weight * direction, iou, track_id, detection_index))

        candidates.sort(reverse=True)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        results: list[TrackedPose] = []

        for _, _, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            track = self._tracks.get(track_id)
            if track is None:
                continue
            detection = detections[detection_index]
            self._correct(track, detection)
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            results.append(TrackedPose(track_id, detection_index, track.bbox_xyxy.copy(), detection.confidence, False))

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            bbox = detection.bbox_xyxy.astype(np.float32, copy=True)
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox_xyxy=bbox,
                velocity_xy=np.zeros(2, dtype=np.float32),
                last_center_xy=bbox_center(bbox),
            )
            matched_tracks.add(track_id)
            results.append(TrackedPose(track_id, detection_index, bbox.copy(), detection.confidence, True))

        stale: list[int] = []
        for track_id, track in self._tracks.items():
            if track_id in matched_tracks:
                continue
            track.age += 1
            track.missed_frames += 1
            track.bbox_xyxy = predicted.get(track_id, track.bbox_xyxy).copy()
            if track.missed_frames > self.max_age:
                stale.append(track_id)
        for track_id in stale:
            del self._tracks[track_id]

        results.sort(key=lambda item: item.pose_index)
        return results

    @staticmethod
    def _predict(track: _TrackState) -> np.ndarray:
        bbox = track.bbox_xyxy.astype(np.float32, copy=True)
        bbox[[0, 2]] += float(track.velocity_xy[0])
        bbox[[1, 3]] += float(track.velocity_xy[1])
        return bbox

    def _correct(self, track: _TrackState, detection: PoseDetection) -> None:
        bbox = detection.bbox_xyxy.astype(np.float32, copy=True)
        center = bbox_center(bbox)
        observed = np.asarray(
            [center[0] - track.last_center_xy[0], center[1] - track.last_center_xy[1]],
            dtype=np.float32,
        )
        alpha = self.velocity_smoothing
        track.velocity_xy = alpha * track.velocity_xy + (1.0 - alpha) * observed
        track.last_center_xy = center
        track.bbox_xyxy = bbox
        track.hits += 1
        track.age += 1
        track.missed_frames = 0


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a[:4])
    bx1, by1, bx2, by2 = map(float, box_b[:4])
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def bbox_center(bbox: np.ndarray) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _direction_consistency(
    velocity: np.ndarray,
    last_center: tuple[float, float],
    detection_center: tuple[float, float],
) -> float:
    velocity_norm = float(np.linalg.norm(velocity))
    movement = np.asarray(
        [detection_center[0] - last_center[0], detection_center[1] - last_center[1]], dtype=np.float32
    )
    movement_norm = float(np.linalg.norm(movement))
    if velocity_norm <= 1e-6 or movement_norm <= 1e-6:
        return 0.5
    cosine = float(np.dot(velocity, movement) / (velocity_norm * movement_norm))
    return (max(-1.0, min(1.0, cosine)) + 1.0) * 0.5

