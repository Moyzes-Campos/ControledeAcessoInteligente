from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from access_control.scrfd import SCRFDFaceDetector


class ScrfdRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.model_path = Path(self.folder.name) / "det_10g.onnx"
        self.engine_path = Path(self.folder.name) / "det_10g_fp16.engine"
        self.model_path.write_bytes(b"model")
        self.engine_path.write_bytes(b"engine")
        self.model = Mock(input_size=None)
        self.cuda_session = Mock()
        self.cuda_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.native_session = Mock()
        self.native_session.get_providers.return_value = ["TensorRTNative"]
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.scrfd_class = stack.enter_context(patch("access_control.scrfd._load_scrfd_class",
                                                     return_value=Mock(return_value=self.model)))
        self.cuda = stack.enter_context(patch("access_control.scrfd.ort.InferenceSession",
                                              return_value=self.cuda_session))
        self.native = stack.enter_context(patch("access_control.tensorrt_session.TensorRTSession",
                                                return_value=self.native_session))

    def test_auto_uses_existing_engine_without_creating_a_cuda_session(self):
        detector = SCRFDFaceDetector(self.model_path, runtime_backend="auto", engine_path=self.engine_path)
        self.assertEqual(detector.providers, ["TensorRTNative"])
        self.cuda.assert_not_called()
        self.native.assert_called_once()

    def test_auto_falls_back_when_engine_is_missing(self):
        detector = SCRFDFaceDetector(self.model_path, runtime_backend="auto",
                                    engine_path=self.engine_path.with_name("missing.engine"))
        self.assertEqual(detector.providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.native.assert_not_called()

    def test_auto_falls_back_on_incompatible_engine_and_reports_reason(self):
        self.native.side_effect = ValueError("incompatible GPU")
        output = io.StringIO()
        with redirect_stdout(output):
            detector = SCRFDFaceDetector(self.model_path, runtime_backend="auto", engine_path=self.engine_path)
        self.assertEqual(detector.runtime_backend, "cuda")
        self.assertIn("incompatible GPU", output.getvalue())
        self.cuda.assert_called_once()

    def test_explicit_tensorrt_propagates_engine_failure(self):
        self.native.side_effect = ValueError("incompatible GPU")
        with self.assertRaisesRegex(ValueError, "incompatible GPU"):
            SCRFDFaceDetector(self.model_path, runtime_backend="tensorrt", engine_path=self.engine_path)
        self.cuda.assert_not_called()

    def test_explicit_cuda_skips_the_engine(self):
        detector = SCRFDFaceDetector(self.model_path, runtime_backend="cuda", engine_path=self.engine_path)
        self.assertEqual(detector.runtime_backend, "cuda")
        self.native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
