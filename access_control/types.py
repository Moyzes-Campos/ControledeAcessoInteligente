from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class PoseDetection:
    bbox_xyxy: np.ndarray
    confidence: float
    keypoints_xyc: np.ndarray


@dataclass(slots=True)
class TrackedPose:
    track_id: int
    pose_index: int
    bbox_xyxy: np.ndarray
    confidence: float
    is_new: bool


@dataclass(slots=True)
class FaceResult:
    bbox_xywh: tuple[int, int, int, int]
    label: str
    score: float
    known: bool
    track_id: int | None = None


@dataclass(slots=True)
class TrackIdentity:
    label: str = "Desconhecido"
    score: float = 0.0
    known: bool = False
    votes: dict[str, int] = field(default_factory=dict)

