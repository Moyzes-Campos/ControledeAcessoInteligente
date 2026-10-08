from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import time

import numpy as np

from .types import FaceResult, TrackedPose


@dataclass(slots=True)
class FaceBatch:
    faces: list[FaceResult]
    tracks: list[TrackedPose]
    frame_index: int
    timings: dict[str, float]


class FaceWorker:
    """One facial job at a time; a busy worker never queues old video frames."""

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-recognition")
        self._future: Future | None = None

    @property
    def busy(self) -> bool:
        # A finished result must be consumed before accepting another job.
        return self._future is not None

    def submit(self, frame: np.ndarray, selected: list[TrackedPose],
               tracks: list[TrackedPose], frame_index: int) -> bool:
        if self.busy or not selected:
            return False
        snapshot = [replace(track, bbox_xyxy=track.bbox_xyxy.copy()) for track in tracks]
        selected_ids = {track.track_id for track in selected}
        selected_snapshot = [track for track in snapshot if track.track_id in selected_ids]
        if not selected_snapshot:
            return False
        self._future = self._executor.submit(
            self._recognize, frame.copy(), selected_snapshot, snapshot, frame_index
        )
        return True

    def poll(self) -> FaceBatch | None:
        if self._future is None or not self._future.done():
            return None
        future, self._future = self._future, None
        return future.result()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _recognize(self, frame, selected, tracks, frame_index) -> FaceBatch:
        started = time.perf_counter()
        faces = self.recognizer.detect_and_recognize(frame, selected)
        timings = dict(self.recognizer.last_timings)
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        return FaceBatch(faces, tracks, frame_index, timings)


class FaceMetrics:
    def __init__(self):
        self.jobs = 0
        self.faces = 0
        self.latest: dict[str, float] = {}
        self._totals: dict[str, float] = {}

    def record(self, faces: list[FaceResult], timings: dict[str, float]) -> None:
        self.jobs += 1
        self.faces += len(faces)
        self.latest = dict(timings)
        for name, value in timings.items():
            self._totals[name] = self._totals.get(name, 0.0) + value

    def summary(self) -> dict:
        return {"jobs_completed": self.jobs, "faces_detected": self.faces,
                "average_ms": {name: round(value / self.jobs, 3)
                               for name, value in self._totals.items()} if self.jobs else {}}


def project_faces(faces: list[FaceResult], snapshot: list[TrackedPose],
                  current: list[TrackedPose]) -> list[FaceResult]:
    """Move the overlay with the same track; identity association uses the snapshot."""
    previous = {track.track_id: track for track in snapshot}
    now = {track.track_id: track for track in current}
    output = []
    for face in faces:
        if face.track_id not in previous or face.track_id not in now:
            continue
        before = previous[face.track_id].bbox_xyxy
        after = now[face.track_id].bbox_xyxy
        old_size = before[2:] - before[:2]
        if np.any(old_size <= 0):
            continue
        scale = (after[2:] - after[:2]) / old_size
        x, y, w, h = face.bbox_xywh
        position = after[:2] + (np.asarray([x, y]) - before[:2]) * scale
        size = np.asarray([w, h]) * scale
        output.append(replace(face, bbox_xywh=tuple(int(round(v)) for v in (*position, *size))))
    return output
