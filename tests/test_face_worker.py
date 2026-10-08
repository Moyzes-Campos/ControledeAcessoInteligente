from concurrent.futures import Future
import threading
import unittest
from unittest.mock import Mock

import numpy as np

from access_control.face_worker import FaceMetrics, FaceWorker, project_faces
from access_control.types import FaceResult, TrackedPose


def track(track_id=1, box=(10, 10, 110, 210)):
    return TrackedPose(track_id, 0, np.asarray(box, dtype=np.float32), .9, False)


class FaceWorkerTests(unittest.TestCase):
    def test_departed_selection_does_not_trigger_full_frame_detection(self):
        recognizer = Mock()
        worker = FaceWorker(recognizer)
        try:
            self.assertFalse(worker.submit(np.zeros((10, 10, 3), np.uint8),
                                           [track(1)], [track(2)], 0))
            self.assertFalse(worker.busy)
            recognizer.detect_and_recognize.assert_not_called()
        finally:
            worker.close()

    def test_busy_worker_does_not_queue_and_preserves_frame_and_track_snapshot(self):
        entered, release = threading.Event(), threading.Event()
        seen = {}

        def recognize(frame, tracks):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test worker was not released")
            seen["pixel"] = int(frame[0, 0, 0])
            seen["box"] = tracks[0].bbox_xyxy.copy()
            return [FaceResult((20, 20, 10, 10), "Moyzes", .8, True)]

        recognizer = Mock(last_timings={"detection_ms": 20.0})
        recognizer.detect_and_recognize.side_effect = recognize
        worker = FaceWorker(recognizer)
        frame = np.zeros((240, 120, 3), np.uint8)
        tracks = [track()]
        try:
            self.assertTrue(worker.submit(frame, tracks, tracks, 42))
            self.assertTrue(entered.wait(3))
            frame[:] = 255
            tracks[0].bbox_xyxy[:] = 0
            self.assertIsNone(worker.poll())
            self.assertFalse(worker.submit(frame, tracks, tracks, 43))
            release.set()
            worker.close()
            batch = worker.poll()
            self.assertEqual(batch.frame_index, 42)
            self.assertEqual(seen["pixel"], 0)
            np.testing.assert_array_equal(seen["box"], [10, 10, 110, 210])
            np.testing.assert_array_equal(batch.tracks[0].bbox_xyxy, [10, 10, 110, 210])
            self.assertFalse(worker.busy)
            recognizer.detect_and_recognize.assert_called_once()
        finally:
            release.set()
            worker.close()

    def test_worker_error_is_reported_to_the_pipeline(self):
        worker = FaceWorker(Mock())
        failed = Future()
        failed.set_exception(RuntimeError("face model failed"))
        worker._future = failed
        try:
            with self.assertRaisesRegex(RuntimeError, "face model failed"):
                worker.poll()
            self.assertFalse(worker.busy)
        finally:
            worker.close()

    def test_overlay_follows_same_track_and_drops_departed_track(self):
        face = FaceResult((20, 30, 10, 20), "Moyzes", .8, True, track_id=1)
        result = project_faces([face], [track()], [track(box=(110, 20, 310, 420))])
        self.assertEqual(result[0].bbox_xywh, (130, 60, 20, 40))
        self.assertEqual(face.bbox_xywh, (20, 30, 10, 20))
        self.assertEqual(project_faces([face], [track()], [track(2)]), [])

    def test_metrics_average_completed_jobs(self):
        metrics = FaceMetrics()
        metrics.record([FaceResult((0, 0, 10, 10), "Desconhecido", .2, False)],
                       {"detection_ms": 20.0, "total_ms": 30.0})
        metrics.record([], {"detection_ms": 10.0, "total_ms": 10.0})
        self.assertEqual(metrics.summary(), {"jobs_completed": 2, "faces_detected": 1,
                                            "average_ms": {"detection_ms": 15.0, "total_ms": 20.0}})


if __name__ == "__main__":
    unittest.main()
