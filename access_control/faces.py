from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
import tempfile
import time

import cv2
import numpy as np

from .types import FaceResult, TrackIdentity, TrackedPose
from .scrfd import SCRFDFaceDetector


RECOGNIZER_THRESHOLDS = {"sface": 0.363, "arcface": 0.40}


class FaceRecognizer:
    def __init__(
        self,
        detector_model: Path,
        recognizer_model: Path,
        gallery_dir: Path,
        threshold: float | None = None,
        detect_score: float = 0.50,
        detector_backend: str = "scrfd",
        rotations: tuple[int, ...] = (0, 90, 180, -90),
        runtime_backend: str = "auto",
        engine_model: Path | None = None,
        recognizer_backend: str = "sface",
        recognizer_engine: Path | None = None,
    ) -> None:
        if recognizer_backend not in RECOGNIZER_THRESHOLDS:
            raise ValueError(f"Reconhecedor facial desconhecido: {recognizer_backend}")
        for model in (detector_model, recognizer_model):
            if not model.exists():
                raise FileNotFoundError(f"Modelo facial nao encontrado: {model}")
        # OpenCV DNN no Windows pode falhar em caminhos com caracteres como "Área".
        detector_runtime = _ascii_runtime_model(detector_model)
        recognizer_runtime = _ascii_runtime_model(recognizer_model)
        self.detector_backend = detector_backend
        if detector_backend == "scrfd":
            engine_path = engine_model or detector_model.with_name(detector_model.stem + "_fp16.engine")
            self.detector = SCRFDFaceDetector(detector_runtime, detect_score, rotations=rotations,
                                             runtime_backend=runtime_backend, engine_path=engine_path)
        elif detector_backend == "yunet":
            self.detector = cv2.FaceDetectorYN.create(
                str(detector_runtime), "", (320, 320), detect_score, 0.3, 5000
            )
        else:
            raise ValueError(f"Detector facial desconhecido: {detector_backend}")
        self.detector_providers = (self.detector.providers if detector_backend == "scrfd"
                                   else ["OpenCV"])
        self.recognizer_backend = recognizer_backend
        if recognizer_backend == "arcface":
            from .arcface import ArcFaceEmbedder
            engine_path = recognizer_engine or recognizer_model.with_name(recognizer_model.stem + "_fp16.engine")
            self.recognizer = ArcFaceEmbedder(recognizer_runtime, runtime_backend, engine_path)
            self.recognizer_providers = self.recognizer.providers
        else:
            self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_runtime), "")
            self.recognizer_providers = ["OpenCV"]
        self.gallery_dir = gallery_dir
        self.threshold = RECOGNIZER_THRESHOLDS[recognizer_backend] if threshold is None else threshold
        self.gallery: dict[str, list[np.ndarray]] = {}
        self.last_timings: dict[str, float] = {}
        self.reload_gallery()

    def reload_gallery(self) -> None:
        self.gallery.clear()
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.gallery_dir.rglob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            # imread on Windows can fail on accented workspace paths.
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            faces = self._detect_raw(image)
            if not faces:
                continue
            face = max(faces, key=lambda item: float(item[2] * item[3]))
            feature = self._feature(image, face)
            label = path.parent.name if path.parent != self.gallery_dir else path.stem
            self.gallery.setdefault(label, []).append(feature)

    def detect_and_recognize(
        self, frame: np.ndarray, tracks: list[TrackedPose] | None = None
    ) -> list[FaceResult]:
        output: list[FaceResult] = []
        started = time.perf_counter()
        faces = self._detect_in_people(frame, tracks) if tracks else self._detect_raw(frame)
        detection_ms = (time.perf_counter() - started) * 1000
        feature_ms = matching_ms = 0.0
        for face in faces:
            x, y, w, h = (int(round(value)) for value in face[:4])
            feature_started = time.perf_counter()
            feature = self._feature(frame, face)
            feature_ms += (time.perf_counter() - feature_started) * 1000
            matching_started = time.perf_counter()
            label, score = self._match(feature)
            matching_ms += (time.perf_counter() - matching_started) * 1000
            known = label is not None and score >= self.threshold
            output.append(FaceResult((x, y, w, h), label if known else "Desconhecido", score, known))
        self.last_timings = {"detection_ms": detection_ms, "feature_ms": feature_ms,
                             "matching_ms": matching_ms,
                             "total_ms": (time.perf_counter() - started) * 1000}
        return output

    def _detect_in_people(self, frame: np.ndarray, tracks: list[TrackedPose]) -> list[np.ndarray]:
        """Busca na pessoa inteira, incluindo faces em cameras giradas."""
        frame_height, frame_width = frame.shape[:2]
        faces: list[np.ndarray] = []
        for track in tracks:
            x1, y1, x2, y2 = map(float, track.bbox_xyxy)
            person_width = x2 - x1
            person_height = y2 - y1
            left = max(0, int(x1 - 0.08 * person_width))
            top = max(0, int(y1 - 0.05 * person_height))
            right = min(frame_width, int(x2 + 0.08 * person_width))
            bottom = min(frame_height, int(y2 + 0.05 * person_height))
            if right - left < 32 or bottom - top < 32:
                continue
            crop = frame[top:bottom, left:right]
            for local_face in self._detect_raw(crop):
                face = local_face.copy()
                face[[0, 4, 6, 8, 10, 12]] += left
                face[[1, 5, 7, 9, 11, 13]] += top
                faces.append(face)
        return faces

    def _detect_raw(self, image: np.ndarray) -> list[np.ndarray]:
        if getattr(self, "detector_backend", "yunet") == "scrfd":
            return self.detector.detect(image)
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []
        return [row.astype(np.float32) for row in faces]

    def _feature(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        if self.recognizer_backend == "arcface":
            return self.recognizer.feature(image, face)
        aligned = self.recognizer.alignCrop(image, face)
        return self.recognizer.feature(aligned)

    def _similarity(self, feature: np.ndarray, reference: np.ndarray) -> float:
        if self.recognizer_backend == "arcface":
            return self.recognizer.match(feature, reference)
        return float(self.recognizer.match(feature, reference, cv2.FaceRecognizerSF_FR_COSINE))

    def _match(self, feature: np.ndarray) -> tuple[str | None, float]:
        best_label: str | None = None
        best_score = -1.0
        for label, features in self.gallery.items():
            score = max(self._similarity(feature, reference) for reference in features)
            if score > best_score:
                best_label, best_score = label, score
        return best_label, best_score


class IdentityMemory:
    def __init__(self, unknown_label: str = "Desconhecido") -> None:
        self.unknown_label = unknown_label
        self._items: dict[int, TrackIdentity] = {}

    def update(self, faces: list[FaceResult], tracks: list[TrackedPose]) -> None:
        for face in faces:
            face.track_id = _associate_face(face, tracks)
            if face.track_id is None or not face.known:
                continue
            identity = self._items.setdefault(face.track_id, TrackIdentity())
            identity.votes[face.label] = identity.votes.get(face.label, 0) + 1
            winner = Counter(identity.votes).most_common(1)[0][0]
            identity.label = winner
            identity.known = True
            if face.label == winner:
                identity.score = max(identity.score, face.score)

    def prune(self, active_track_ids: set[int]) -> None:
        for track_id in list(self._items):
            if track_id not in active_track_ids:
                del self._items[track_id]

    def get(self, track_id: int) -> TrackIdentity:
        return self._items.get(track_id, TrackIdentity(label=self.unknown_label))


def _associate_face(face: FaceResult, tracks: list[TrackedPose]) -> int | None:
    x, y, w, h = face.bbox_xywh
    center_x, center_y = x + w / 2.0, y + h / 2.0
    candidates: list[tuple[float, int]] = []
    for track in tracks:
        x1, y1, x2, y2 = map(float, track.bbox_xyxy)
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            # Prefere a caixa cuja regiao superior contem melhor o rosto.
            upper_center_y = y1 + 0.25 * (y2 - y1)
            candidates.append((abs(center_y - upper_center_y), track.track_id))
    return min(candidates)[1] if candidates else None


def _ascii_runtime_model(source: Path) -> Path:
    try:
        str(source.resolve()).encode("ascii")
        return source
    except UnicodeEncodeError:
        runtime_dir = Path(tempfile.gettempdir()) / "cargil_access_control_models"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        destination = runtime_dir / source.name
        if not destination.exists() or destination.stat().st_size != source.stat().st_size:
            shutil.copyfile(source, destination)
        return destination

