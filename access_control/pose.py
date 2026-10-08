from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .types import PoseDetection


class PoseDetector:
    def __init__(self, model_path: Path, device: str = "0", image_size: int = 640, confidence: float = 0.25) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")
        task = "pose" if "pose" in model_path.stem.lower() else "detect"
        self.model = YOLO(str(model_path), task=task)
        print(f"Modelo YOLO: {model_path.resolve()} | tarefa: {task} | device: {device}", flush=True)
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.last_timings: dict[str, float] = {}

    def detect(self, frame: np.ndarray) -> list[PoseDetection]:
        result = self.model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=[0],
            device=self.device,
            verbose=False,
        )[0]
        self.last_timings = dict(result.speed)
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        if result.keypoints is None:
            return [PoseDetection(box.astype(np.float32), float(score),
                                  np.empty((0, 3), dtype=np.float32))
                    for box, score in zip(boxes, scores)]
        points_xy = result.keypoints.xy.detach().cpu().numpy()
        points_conf = result.keypoints.conf
        if points_conf is None:
            confidence = np.ones(points_xy.shape[:2], dtype=np.float32)
        else:
            confidence = points_conf.detach().cpu().numpy()
        return [
            PoseDetection(
                bbox_xyxy=box.astype(np.float32),
                confidence=float(score),
                keypoints_xyc=np.column_stack((xy, conf)).astype(np.float32),
            )
            for box, score, xy, conf in zip(boxes, scores, points_xy, confidence)
        ]

