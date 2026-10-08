from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# Template de 5 pontos do InsightFace para recortes 112x112 (olhos, nariz, boca).
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)
INPUT_SIZE = 112


class ArcFaceEmbedder:
    """InsightFace ArcFace (w600k_r50) que gera embeddings normalizados de 512 valores."""

    def __init__(
        self,
        model_path: Path,
        runtime_backend: str = "auto",
        engine_path: Path | None = None,
        cpu_threads: int = 2,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo ArcFace nao encontrado: {model_path}")
        # Importar torch antes pre-carrega as DLLs CUDA/cuDNN usadas pelo ORT.
        import torch
        _ = torch.cuda.is_available()
        if runtime_backend not in {"auto", "cuda", "tensorrt"}:
            raise ValueError(f"Runtime facial desconhecido: {runtime_backend}")
        session = None
        if runtime_backend != "cuda":
            if engine_path is not None and engine_path.exists():
                try:
                    from .tensorrt_session import TensorRTSession
                    session = TensorRTSession(engine_path, model_path, (INPUT_SIZE, INPUT_SIZE))
                except Exception as error:
                    if runtime_backend == "tensorrt":
                        raise
                    print(f"Engine ArcFace indisponivel: {error}. Usando ONNX/CUDA.", flush=True)
            elif runtime_backend == "tensorrt":
                raise FileNotFoundError(f"Engine ArcFace nao encontrada: {engine_path}")
        if session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, cpu_threads)
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            session = ort.InferenceSession(str(model_path), sess_options=options,
                                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.providers = session.get_providers()

    def feature(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(align_face(image, face), 1.0 / 127.5, (INPUT_SIZE, INPUT_SIZE),
                                     (127.5, 127.5, 127.5), swapRB=True)
        embedding = self.session.run([self.output_name], {self.input_name: blob})[0][0]
        return (embedding / max(float(np.linalg.norm(embedding)), 1e-12)).astype(np.float32)

    @staticmethod
    def match(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.dot(first, second))


def align_face(image: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Recorte 112x112 alinhado pelos 5 pontos no layout de 15 valores (SCRFD/YuNet)."""
    landmarks = np.asarray(face[4:14], dtype=np.float64).reshape(5, 2)
    return cv2.warpAffine(image, similarity_transform(landmarks, ARCFACE_TEMPLATE),
                          (INPUT_SIZE, INPUT_SIZE), borderValue=0.0)


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Similaridade (escala, rotacao, translacao) por minimos quadrados, como o norm_crop."""
    x, y = source[:, 0], source[:, 1]
    ones, zeros = np.ones_like(x), np.zeros_like(x)
    system = np.vstack((np.column_stack((x, -y, ones, zeros)),
                        np.column_stack((y, x, zeros, ones))))
    values = np.concatenate((target[:, 0], target[:, 1])).astype(np.float64)
    (a, b, tx, ty), *_ = np.linalg.lstsq(system, values, rcond=None)
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)
