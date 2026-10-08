from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from access_control.arcface import ARCFACE_TEMPLATE, ArcFaceEmbedder, similarity_transform
from access_control.faces import FaceRecognizer


class AlignmentTests(unittest.TestCase):
    def test_rotated_and_scaled_landmarks_map_back_to_the_template(self):
        angle = np.deg2rad(90)
        rotation = 2.5 * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        landmarks = ARCFACE_TEMPLATE @ rotation.T + np.array([400.0, 120.0])
        matrix = similarity_transform(landmarks, ARCFACE_TEMPLATE)
        mapped = landmarks @ matrix[:, :2].T + matrix[:, 2]
        np.testing.assert_allclose(mapped, ARCFACE_TEMPLATE, atol=1e-4)


class ArcFaceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.model_path = Path(self.folder.name) / "w600k_r50.onnx"
        self.engine_path = Path(self.folder.name) / "w600k_r50_fp16.engine"
        self.model_path.write_bytes(b"model")
        self.engine_path.write_bytes(b"engine")
        self.cuda_session = self._session(["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.native_session = self._session(["TensorRTNative"])
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.cuda = stack.enter_context(patch("access_control.arcface.ort.InferenceSession",
                                              return_value=self.cuda_session))
        self.native = stack.enter_context(patch("access_control.tensorrt_session.TensorRTSession",
                                                return_value=self.native_session))

    @staticmethod
    def _session(providers):
        session = Mock()
        session.get_providers.return_value = providers
        session.get_inputs.return_value = [Mock()]
        session.get_outputs.return_value = [Mock()]
        return session

    def test_auto_uses_existing_engine_at_112_pixels(self):
        embedder = ArcFaceEmbedder(self.model_path, "auto", self.engine_path)
        self.assertEqual(embedder.providers, ["TensorRTNative"])
        self.assertEqual(self.native.call_args.args[2], (112, 112))
        self.cuda.assert_not_called()

    def test_auto_falls_back_on_incompatible_engine_and_reports_reason(self):
        self.native.side_effect = ValueError("incompatible GPU")
        output = io.StringIO()
        with redirect_stdout(output):
            embedder = ArcFaceEmbedder(self.model_path, "auto", self.engine_path)
        self.assertEqual(embedder.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertIn("incompatible GPU", output.getvalue())

    def test_explicit_tensorrt_requires_the_engine(self):
        with self.assertRaises(FileNotFoundError):
            ArcFaceEmbedder(self.model_path, "tensorrt", self.engine_path.with_name("missing.engine"))
        self.cuda.assert_not_called()

    def test_explicit_cuda_skips_the_engine(self):
        ArcFaceEmbedder(self.model_path, "cuda", self.engine_path)
        self.native.assert_not_called()

    def test_feature_is_unit_length(self):
        self.native_session.run.return_value = [np.array([[3.0, 4.0]], dtype=np.float32)]
        embedder = ArcFaceEmbedder(self.model_path, "auto", self.engine_path)
        face = np.concatenate(([0, 0, 112, 112], ARCFACE_TEMPLATE.reshape(-1), [.9])).astype(np.float32)
        feature = embedder.feature(np.zeros((200, 200, 3), np.uint8), face)
        np.testing.assert_allclose(feature, [0.6, 0.8], atol=1e-6)
        self.assertEqual(self.native_session.run.call_args.args[1][embedder.input_name].shape,
                         (1, 3, 112, 112))


class RecognizerSelectionTests(unittest.TestCase):
    def test_arcface_matches_with_dot_product_of_normalized_embeddings(self):
        subject = FaceRecognizer.__new__(FaceRecognizer)
        subject.recognizer_backend = "arcface"
        subject.recognizer = ArcFaceEmbedder.__new__(ArcFaceEmbedder)
        subject.gallery = {"Moyzes": [np.array([1.0, 0.0], np.float32)],
                           "Outro": [np.array([0.0, 1.0], np.float32)]}
        label, score = subject._match(np.array([0.8, 0.6], np.float32))
        self.assertEqual(label, "Moyzes")
        self.assertAlmostEqual(score, 0.8, places=6)

    def test_unknown_recognizer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Reconhecedor"):
            FaceRecognizer(Path("det.onnx"), Path("rec.onnx"), Path("faces"), recognizer_backend="facenet")
