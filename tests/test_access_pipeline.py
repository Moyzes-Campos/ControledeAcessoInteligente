import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np

from access_control.pipeline import PipelineConfig, run_pipeline
from access_control.face_worker import FaceBatch
from access_control.timing import FrameRateLimiter, FrameRateMeter
from access_control.types import FaceResult, PoseDetection, TrackedPose


class AccessPipelineTests(unittest.TestCase):
    def test_live_camera_pause_services_gui_and_quits_without_running_detection_again(self):
        for quit_key in (27, ord('q')):
            with self.subTest(quit_key=quit_key), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                frame = np.zeros((200, 320, 3), np.uint8)
                capture = MagicMock()
                capture.get.side_effect = lambda key: {
                    cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 200,
                    cv2.CAP_PROP_FPS: 20,
                }.get(key, 0)
                callbacks = []

                def read(on_wait=None):
                    if not callbacks:
                        callbacks.append(None)
                        return True, frame
                    self.assertIsNotNone(on_wait)
                    callbacks.append(on_wait)
                    self.assertTrue(on_wait())
                    self.assertFalse(on_wait())
                    return False, None

                capture.read.side_effect = read
                detector = Mock()
                detector.detect.return_value = []
                recognizer = Mock(gallery={}, detector_providers=['TensorRTNative'],
                                  recognizer_providers=['OpenCV'], threshold=0.4)
                config = PipelineConfig('rtsp://test/stream', root / 'outputs', root / 'det.engine',
                                        root / 'face.onnx', root / 'sface.onnx', root / 'faces',
                                        database=root / 'test.sqlite3', line_config=None,
                                        esp_url=None, show=True)
                closed = []
                capture.release.side_effect = lambda: closed.append('capture')
                with patch('access_control.pipeline.open_capture', return_value=capture), \
                     patch('access_control.pipeline.PoseDetector', return_value=detector), \
                     patch('access_control.pipeline.FaceRecognizer', return_value=recognizer), \
                     patch('access_control.pipeline.create_display_window'), \
                     patch('access_control.pipeline.cv2.imshow'), \
                     patch('access_control.pipeline.cv2.destroyAllWindows', side_effect=lambda: closed.append('window')), \
                     patch('access_control.pipeline.poll_display_key', side_effect=[255, 255, quit_key]):
                    summary = run_pipeline(config)
                self.assertEqual(summary['frames_processed'], 1)
                detector.detect.assert_called_once()
                recognizer.detect_and_recognize.assert_not_called()
                self.assertEqual(closed, ['window', 'capture'])
                rows = (root / 'outputs' / 'eventos.jsonl').read_text(encoding='utf-8').splitlines()
                self.assertEqual(len(rows), 1)

    def test_gui_without_recording_draws_on_preview_and_keeps_original_analysis_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = np.zeros((1520, 2688, 3), np.uint8)
            capture = MagicMock()
            capture.get.side_effect = lambda key: {
                cv2.CAP_PROP_FRAME_WIDTH: 2688, cv2.CAP_PROP_FRAME_HEIGHT: 1520,
                cv2.CAP_PROP_FPS: 20,
            }.get(key, 0)
            capture.read.side_effect = [(True, frame), (False, None)]
            detector = Mock()
            detector.detect.return_value = []
            recognizer = Mock(gallery={}, detector_providers=['TensorRTNative'],
                              recognizer_providers=['OpenCV'], threshold=0.4)
            config = PipelineConfig('test.mp4', root / 'outputs', root / 'det.engine',
                                    root / 'face.onnx', root / 'sface.onnx', root / 'faces',
                                    database=root / 'test.sqlite3', line_config=None, esp_url=None,
                                    show=True)
            with patch('access_control.pipeline.open_capture', return_value=capture), \
                 patch('access_control.pipeline.PoseDetector', return_value=detector), \
                 patch('access_control.pipeline.FaceRecognizer', return_value=recognizer), \
                 patch('access_control.pipeline.draw_scene', side_effect=lambda image, *args, **kw: image) as draw, \
                 patch('access_control.pipeline.draw_access_overlay') as overlay, \
                 patch('access_control.pipeline.create_display_window'), \
                 patch('access_control.pipeline.cv2.imshow') as show, \
                 patch('access_control.pipeline.poll_display_key', return_value=255):
                summary = run_pipeline(config)
            self.assertIs(detector.detect.call_args.args[0], frame)
            preview = draw.call_args.args[0]
            self.assertLessEqual(max(preview.shape[:2]), 1280)
            self.assertIs(show.call_args.args[1], preview)
            expected_scale = (preview.shape[1] / 2688, preview.shape[0] / 1520)
            self.assertEqual(draw.call_args.kwargs['coordinate_scale'], expected_scale)
            self.assertEqual(overlay.call_args.kwargs['coordinate_scale'], expected_scale)
            self.assertEqual(summary['resolution'], [2688, 1520])
            self.assertIsNone(summary['output_video'])

    def test_gui_preview_keeps_detection_and_video_at_original_resolution_and_logs_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            frame = np.zeros((1520, 2688, 3), np.uint8)
            capture = MagicMock()
            capture.get.side_effect = lambda key: {
                cv2.CAP_PROP_FRAME_WIDTH: 2688, cv2.CAP_PROP_FRAME_HEIGHT: 1520,
                cv2.CAP_PROP_FPS: 20,
            }.get(key, 0)
            capture.read.side_effect = [(True, frame), (False, None)]
            capture.get_stats.return_value = {'received_fps': 20.0, 'frames_dropped': 0}
            detector = Mock(last_timings={'preprocess': 2.0, 'inference': 10.0, 'postprocess': 1.0})
            detector.detect.return_value = []
            recognizer = Mock(gallery={}, detector_providers=['TensorRTNative'],
                              recognizer_providers=['OpenCV'], threshold=0.4)
            writer = Mock()
            config = PipelineConfig('test.mp4', root / 'outputs', root / 'det.engine',
                                    root / 'face.onnx', root / 'sface.onnx', root / 'faces',
                                    database=root / 'test.sqlite3', line_config=None, esp_url=None,
                                    show=True, save_video=True)
            with patch('access_control.pipeline.open_capture', return_value=capture), \
                 patch('access_control.pipeline.PoseDetector', return_value=detector), \
                 patch('access_control.pipeline.FaceRecognizer', return_value=recognizer), \
                 patch('access_control.pipeline._create_writer', return_value=writer), \
                 patch('access_control.pipeline.create_display_window'), \
                 patch('access_control.pipeline.cv2.imshow') as show, \
                 patch('access_control.pipeline.poll_display_key', return_value=255):
                summary = run_pipeline(config)
            self.assertIs(detector.detect.call_args.args[0], frame)
            self.assertEqual(writer.write.call_args.args[0].shape, frame.shape)
            self.assertLessEqual(max(show.call_args.args[1].shape[:2]), 1280)
            self.assertEqual(summary['frames_processed'], 1)
            self.assertEqual(summary['capture']['received_fps'], 20.0)
            self.assertEqual(summary['average_yolo_timings_ms'], detector.last_timings)
            timings = summary['average_timings_ms']
            stages = ['person_detection', 'tracking', 'limiter_wait', 'capture_wait',
                      'rotation', 'access_and_faces', 'drawing', 'video_write', 'display', 'event_write']
            self.assertTrue(all(timings[name] >= 0 for name in stages))
            # The old `frame` field overlaps analysis; the new cycle includes each actual stage once.
            self.assertGreaterEqual(timings['cycle'] + .1, sum(timings[name] for name in stages))
            event = json.loads((root / 'outputs' / 'eventos.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(event['capture']['received_fps'], 20.0)
            self.assertIn('display', event['timings_ms'])
            writer.release.assert_called_once()

    def test_empty_live_scene_limits_reads_and_reports_real_fps_without_face_jobs(self):
        class Clock:
            now = 0.0

            def __call__(self):
                return self.now

            def sleep(self, seconds):
                self.now += seconds

        for configured_fps, expected_fps in ((0, 20), (10, 10)):
            with self.subTest(max_fps=configured_fps), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                clock = Clock()
                reads = []
                capture = MagicMock()
                capture.get.side_effect = lambda key: {
                    cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 200,
                    cv2.CAP_PROP_FPS: 20,
                }.get(key, 0)

                def read():
                    if len(reads) == 4:
                        return False, None
                    reads.append(clock.now)
                    return True, np.zeros((200, 320, 3), np.uint8)

                def detect(_frame):
                    clock.now += .02
                    return []

                capture.read.side_effect = read
                detector = Mock()
                detector.detect.side_effect = detect
                recognizer = Mock(gallery={}, detector_providers=["CUDAExecutionProvider"],
                                  recognizer_providers=["OpenCV"], threshold=0.4)
                config = PipelineConfig("rtsp://test/stream", root / "outputs", root / "det.engine",
                                        root / "face.onnx", root / "sface.onnx", root / "faces",
                                        database=root / "test.sqlite3", line_config=None,
                                        esp_url=None, face_interval=1, max_fps=configured_fps)
                with patch("access_control.pipeline.open_capture", return_value=capture), \
                     patch("access_control.pipeline.PoseDetector", return_value=detector), \
                     patch("access_control.pipeline.FaceRecognizer", return_value=recognizer), \
                     patch("access_control.pipeline.FrameRateLimiter", return_value=FrameRateLimiter(
                         expected_fps, clock=clock, sleep=clock.sleep)), \
                     patch("access_control.pipeline.FrameRateMeter", return_value=FrameRateMeter(clock=clock)), \
                     patch("access_control.pipeline.draw_scene", side_effect=lambda frame, *args, **kwargs: frame) as draw:
                    summary = run_pipeline(config)
                self.assertEqual(summary["processing_fps_limit"], expected_fps)
                self.assertEqual(summary["face_processing"]["jobs_completed"], 0)
                recognizer.detect_and_recognize.assert_not_called()
                for first, second in zip(reads, reads[1:]):
                    self.assertAlmostEqual(second - first, 1.0 / expected_fps)
                self.assertAlmostEqual(draw.call_args.args[5], expected_fps)
                events = [json.loads(row) for row in (root / "outputs" / "eventos.jsonl")
                          .read_text(encoding="utf-8").splitlines()]
                self.assertEqual(events[-1]["processing_fps"], expected_fps)

    def test_live_delayed_face_uses_original_boxes_and_does_not_identify_a_new_id(self):
        for second_id in (1, 2):
            with self.subTest(second_id=second_id), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                original = TrackedPose(1, 0, np.array([20, 20, 80, 150]), .9, True)
                # The person has moved; the old face is outside the current box.
                current = TrackedPose(second_id, 0, np.array([180, 20, 240, 150]), .9, False)
                tracks_by_frame = [[original], [current]]

                class Tracker:
                    active_track_ids = set()

                    def update(self, _detections):
                        tracks = tracks_by_frame.pop(0)
                        self.active_track_ids = {item.track_id for item in tracks}
                        return tracks

                capture = MagicMock()
                capture.get.side_effect = lambda key: {
                    cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 200,
                    cv2.CAP_PROP_FPS: 20,
                }.get(key, 0)
                capture.read.side_effect = [(True, np.zeros((200, 320, 3), np.uint8)),
                                            (True, np.zeros((200, 320, 3), np.uint8)),
                                            (False, None)]
                detector = Mock()
                detector.detect.side_effect = [[PoseDetection(item.bbox_xyxy, .9, np.empty((0, 3)))]
                                              for item in (original, current)]
                recognizer = Mock(gallery={"Moyzes": []}, detector_providers=["CUDAExecutionProvider"],
                                  recognizer_providers=["OpenCV"], threshold=0.4)
                worker = Mock(busy=False)
                worker.submit.return_value = True
                worker.poll.side_effect = [None, FaceBatch(
                    [FaceResult((30, 30, 15, 15), "Moyzes", .8, True)],
                    [original], 0, {"total_ms": 100.0}), None]
                config = PipelineConfig("rtsp://test/stream", root / "outputs", root / "det.engine",
                                        root / "face.onnx", root / "sface.onnx", root / "faces",
                                        database=root / "acessos.sqlite3", line_config=None,
                                        esp_url=None, face_interval=1)
                with patch("access_control.pipeline.open_capture", return_value=capture), \
                     patch("access_control.pipeline.PoseDetector", return_value=detector), \
                     patch("access_control.pipeline.FaceRecognizer", return_value=recognizer), \
                     patch("access_control.pipeline.OCSortPersonTracker", return_value=Tracker()), \
                     patch("access_control.pipeline.FaceWorker", return_value=worker):
                    summary = run_pipeline(config)
                events = [json.loads(row) for row in (root / "outputs" / "eventos.jsonl")
                          .read_text(encoding="utf-8").splitlines()]
                self.assertFalse(events[0]["tracks"][0]["known"])
                self.assertEqual(events[1]["tracks"][0]["known"], second_id == 1)
                self.assertEqual(summary["face_processing"]["mode"], "async")
                self.assertEqual(summary["face_processing"]["jobs_completed"], 1)
                recognizer.detect_and_recognize.assert_not_called()
                worker.close.assert_called_once()
                capture.release.assert_called_once()

    def test_pipeline_records_crossings_and_drives_signals_with_new_tracking(self):
        positions = [(1, 80), (1, 120), None, (9, 120), (9, 80), (10, 80), (10, 120)]
        tracks_by_frame = []
        for position in positions:
            if position is None:
                tracks_by_frame.append([])
                continue
            track_id, y = position
            tracks_by_frame.append([
                TrackedPose(track_id, 0, np.array([80, y - 25, 120, y + 25]), .9, False)
            ])

        class Tracker:
            active_track_ids = set()

            def update(self, _detections):
                tracks = tracks_by_frame.pop(0)
                self.active_track_ids = {track.track_id for track in tracks}
                return tracks

        def recognize(_frame, tracks):
            return [FaceResult((85, int(track.bbox_xyxy[1]) + 5, 15, 15), "Moyzes", .8, True)
                    for track in tracks if track.track_id != 10]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            line = root / "linha.json"
            line.write_text(json.dumps({"line_normalized": [0, .5, 1, .5],
                                        "hysteresis_pixels": 2}), encoding="utf-8")
            capture = MagicMock()
            capture.get.side_effect = lambda key: {
                cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 200,
                cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_COUNT: len(positions),
            }.get(key, 0)
            capture.read.side_effect = [(True, np.zeros((200, 320, 3), np.uint8))
                                       for _ in positions] + [(False, None)]
            detector = MagicMock()
            detector.detect.side_effect = [
                [PoseDetection(tracks[0].bbox_xyxy, .9, np.empty((0, 3)))] if tracks else []
                for tracks in tracks_by_frame
            ]
            recognizer = MagicMock()
            recognizer.gallery = {"Moyzes": []}
            recognizer.detect_and_recognize.side_effect = recognize
            recognizer.recognizer_providers = ["OpenCV"]
            recognizer.threshold = 0.4
            esp = MagicMock()
            esp.status = {"url": "http://esp-test", "connected": True, "last_error": None}
            config = PipelineConfig("test.mp4", root / "outputs", root / "det.engine",
                                    root / "face.onnx", root / "sface.onnx", root / "faces",
                                    database=root / "acessos.sqlite3", line_config=line,
                                    esp_url="http://esp-test", face_interval=1)
            with patch("access_control.pipeline.open_capture", return_value=capture), \
                 patch("access_control.pipeline.PoseDetector", return_value=detector), \
                 patch("access_control.pipeline.FaceRecognizer", return_value=recognizer), \
                 patch("access_control.pipeline.OCSortPersonTracker", return_value=Tracker()), \
                 patch("access_control.pipeline.EspGpioClient", return_value=esp):
                summary = run_pipeline(config)

            self.assertEqual((summary["entries"], summary["exits"], summary["unknown_crossings"]),
                             (2, 1, 1))
            self.assertEqual(summary["frames_processed"], 7)
            self.assertIsNone(summary["output_video"])
            self.assertFalse((root / "outputs" / "resultado.mp4").exists())
            with closing(sqlite3.connect(root / "acessos.sqlite3")) as connection:
                visits = connection.execute(
                    "SELECT nome, tracking_entrada, tracking_saida, tempo_permanencia_segundos FROM acessos"
                ).fetchall()
                self.assertEqual(visits, [("Moyzes", 1, 9, .3)])
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM cruzamentos").fetchone()[0], 3)
            events = [json.loads(row) for row in (root / "outputs" / "eventos.jsonl")
                      .read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(events), 7)
            self.assertEqual([call.args[0] for call in esp.update.call_args_list][1::2],
                             ["green", "green", "orange", "green", "green", "red", "red"])
            self.assertEqual(esp.update.call_args_list[0].args[0], "red")
            self.assertTrue(events[-1]["access"]["alarm"])
            esp.close.assert_called_once()
            capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
