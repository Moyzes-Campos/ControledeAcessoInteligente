from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class SCRFDFaceDetector:
    """SCRFD adapter that returns the 15-value layout expected by SFace."""

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.60,
        input_size: tuple[int, int] = (640, 640),
        rotations: tuple[int, ...] = (0, 90, 180, -90),
        cpu_threads: int = 2,
        runtime_backend: str = "cuda",
        engine_path: Path | None = None,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo SCRFD nao encontrado: {model_path}")
        scrfd_class = _load_scrfd_class()
        # Importing torch first preloads the CUDA 12.1/cuDNN 8 DLLs already
        # shipped with this project's environment, as recommended by ORT.
        import torch
        _ = torch.cuda.is_available()
        if runtime_backend not in {"auto", "cuda", "tensorrt"}:
            raise ValueError(f"Runtime facial desconhecido: {runtime_backend}")
        session = None
        if runtime_backend != "cuda":
            if engine_path is not None and engine_path.exists():
                try:
                    from .tensorrt_session import TensorRTSession
                    session = TensorRTSession(engine_path, model_path, input_size)
                except Exception as error:
                    if runtime_backend == "tensorrt":
                        raise
                    print(f"Engine facial indisponivel: {error}. Usando ONNX/CUDA.", flush=True)
            elif runtime_backend == "tensorrt":
                raise FileNotFoundError(f"Engine SCRFD nao encontrada: {engine_path}; execute exportar_scrfd_engine.py")
        if session is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, cpu_threads)
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        self.model = scrfd_class(model_file=str(model_path), session=session)
        if self.model.input_size is None:
            self.model.prepare(0, det_thresh=score_threshold, input_size=input_size)
        else:
            self.model.prepare(0, det_thresh=score_threshold)
        self.rotations = rotations
        self.providers = session.get_providers()
        self.runtime_backend = "tensorrt" if self.providers == ["TensorRTNative"] else "cuda"

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        output: list[np.ndarray] = []
        for angle in self.rotations:
            rotated = _rotate(image, angle)
            boxes, landmarks = self.model.detect(rotated)
            if landmarks is None:
                continue
            for box, points in zip(boxes, landmarks):
                x1, y1, x2, y2, score = box
                row = np.concatenate((
                    np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float32),
                    points.reshape(-1).astype(np.float32),
                    np.asarray([score], dtype=np.float32),
                ))
                output.append(_restore_detection(row, angle, image.shape[1], image.shape[0]))
        return _nms(output, 0.4)


def _rotate(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    return image


def _restore_detection(row: np.ndarray, angle: int, width: int, height: int) -> np.ndarray:
    x, y, w, h = row[:4]
    corners = np.asarray(((x, y), (x + w, y + h)), dtype=np.float32)
    landmarks = row[4:14].reshape(5, 2)
    corners = _restore_points(corners, angle, width, height)
    landmarks = _restore_points(landmarks, angle, width, height)
    low, high = corners.min(axis=0), corners.max(axis=0)
    return np.concatenate((low, high - low, landmarks.reshape(-1), row[-1:])).astype(np.float32)


def _restore_points(points: np.ndarray, angle: int, width: int, height: int) -> np.ndarray:
    restored = points.copy()
    if angle == 90:
        restored[:, 0], restored[:, 1] = points[:, 1], height - 1 - points[:, 0]
    elif angle == -90:
        restored[:, 0], restored[:, 1] = width - 1 - points[:, 1], points[:, 0]
    elif angle == 180:
        restored[:, 0], restored[:, 1] = width - 1 - points[:, 0], height - 1 - points[:, 1]
    return restored


def _nms(rows: list[np.ndarray], threshold: float) -> list[np.ndarray]:
    if not rows:
        return []
    pending = sorted(rows, key=lambda row: float(row[-1]), reverse=True)
    kept: list[np.ndarray] = []
    while pending:
        best = pending.pop(0)
        kept.append(best)
        pending = [row for row in pending if _iou(best, row) <= threshold]
    return kept


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    ax1, ay1, aw, ah = first[:4]
    bx1, by1, bw, bh = second[:4]
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 0 else 0.0


def _load_scrfd_class():
    """Load only InsightFace's SCRFD module, avoiding its optional app stack."""
    package = importlib.util.find_spec("insightface")
    if package is None or not package.submodule_search_locations:
        raise RuntimeError("Dependencia insightface nao instalada; consulte requirements.txt")
    source = Path(next(iter(package.submodule_search_locations))) / "model_zoo" / "scrfd.py"
    spec = importlib.util.spec_from_file_location("_insightface_scrfd_runtime", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar SCRFD de {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SCRFD
