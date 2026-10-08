import threading
import unittest
from unittest.mock import patch

import cv2

from access_control.capture import LatestFrameCapture, open_capture


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return 30

    def set(self, *args):
        return True

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame

    def release(self):
        self.released = True


class CaptureTests(unittest.TestCase):
    @staticmethod
    def waiting_capture():
        # Seed a consumed frame so a timeout must not return it again.
        capture = LatestFrameCapture.__new__(LatestFrameCapture)
        capture._condition = threading.Condition()
        capture._stop = threading.Event()
        capture._sequence = capture._consumed = 1
        capture._latest = 1
        capture._delivered_frames = 1
        capture._dropped_frames = 0
        capture._latest_frame_info = {'sequence': 1}
        capture._delivered_frame_info = {'sequence': 1}
        return capture

    def test_wait_callback_can_quit_without_reprocessing_the_last_frame(self):
        capture = self.waiting_capture()
        self.assertEqual(capture.read(on_wait=lambda: False), (False, None))
        self.assertEqual(capture._delivered_frames, 1)
        self.assertEqual(capture._consumed, 1)
        self.assertEqual(capture._dropped_frames, 0)

    def test_wait_callback_releases_capture_lock_and_receives_new_frame(self):
        capture = self.waiting_capture()
        published = threading.Event()
        workers = []

        def publish():
            with capture._condition:
                capture._sequence = 2
                capture._latest = 2
                capture._latest_frame_info = {'sequence': 2}
                capture._condition.notify_all()
            published.set()

        def service_gui():
            worker = threading.Thread(target=publish, daemon=True)
            workers.append(worker)
            worker.start()
            self.assertTrue(published.wait(1), 'GUI callback held the capture lock')
            return True

        try:
            self.assertEqual(capture.read(on_wait=service_gui), (True, 2))
            self.assertEqual(capture._delivered_frames, 2)
            self.assertEqual(capture._delivered_frame_info['sequence'], 2)
        finally:
            for worker in workers:
                worker.join(timeout=1)

    def test_native_read_stall_on_discarded_iframe_is_still_reported(self):
        class Clock:
            now = 100.0

            def __call__(self):
                return self.now

        clock = Clock()

        class TypedCapture(FakeCapture):
            frame_type = 0

            def read(self):
                ok, frame = super().read()
                if ok:
                    self.frame_type = 73 if frame == 1 else 80
                    clock.now += .3 if frame == 1 else .01
                return ok, frame

            def get(self, key):
                return self.frame_type if key == cv2.CAP_PROP_FRAME_TYPE else super().get(key)

        native = TypedCapture([1, 2, 3])
        with patch.object(LatestFrameCapture, '_connect', return_value=native), \
             patch('access_control.capture.time.perf_counter', side_effect=clock):
            capture = LatestFrameCapture('rtsp://example/live')
            try:
                with capture._condition:
                    self.assertTrue(capture._condition.wait_for(lambda: capture._sequence == 3, timeout=1))
                self.assertEqual(capture.read(), (True, 3))
                stats = capture.get_stats()
                self.assertEqual(stats['frames_dropped'], 2)
                self.assertEqual(stats['native_read_stalls_ge_200ms'], {'I': 1})
                self.assertEqual(stats['delivered_frame']['sequence'], 3)
                self.assertEqual(stats['delivered_frame']['frame_type'], 'P')
                self.assertEqual(stats['delivered_frame']['native_read_ms'], 10.0)
                self.assertEqual(stats['delivered_frame']['arrival_interval_ms'], 10.0)
            finally:
                capture.release()

    def test_file_keeps_all_frames_in_order(self):
        fake = FakeCapture([1, 2, 3])
        with patch('access_control.capture.cv2.VideoCapture', return_value=fake):
            capture = open_capture('example.mp4')
            self.assertEqual([capture.read()[1] for _ in range(3)], [1, 2, 3])
            self.assertFalse(capture.read()[0])

    def test_live_discards_old_frames_and_reconnects(self):
        first = FakeCapture([1, 2, 3])
        second = FakeCapture([4])
        connected = threading.Event()

        def connect(subject):
            if not first.released:
                return first
            connected.set()
            return second

        with patch.object(LatestFrameCapture, '_connect', connect):
            capture = LatestFrameCapture('rtsp://example/live')
            try:
                self.assertEqual(capture.read(), (True, 3))
                self.assertTrue(connected.wait(3))
                self.assertEqual(capture.read(), (True, 4))
                stats = capture.get_stats()
                self.assertEqual(stats['frames_received'], 4)
                self.assertEqual(stats['frames_delivered'], 2)
                self.assertEqual(stats['frames_dropped'], 2)
                self.assertGreaterEqual(stats['latest_frame_age_ms'], 0)
            finally:
                capture.release()
            self.assertTrue(first.released)
            self.assertTrue(second.released)
            self.assertFalse(capture._thread.is_alive())

    def test_rtsp_uses_ffmpeg_with_timeouts_and_tcp(self):
        with patch.dict('os.environ', {}, clear=True), patch(
            'access_control.capture.cv2.VideoCapture', return_value=FakeCapture([1])
        ) as factory:
            capture = LatestFrameCapture('rtsp://example/live')
            capture.release()
            self.assertEqual(factory.call_args.args[1], cv2.CAP_FFMPEG)
            self.assertIn(cv2.CAP_PROP_READ_TIMEOUT_MSEC, factory.call_args.args[2])
            parameters = dict(zip(factory.call_args.args[2][::2], factory.call_args.args[2][1::2]))
            self.assertEqual(parameters[cv2.CAP_PROP_N_THREADS], 1)
            import os
            self.assertEqual(os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'], 'rtsp_transport;tcp')

    def test_decoder_automatic_mode_can_be_restored(self):
        with patch('access_control.capture.cv2.VideoCapture', return_value=FakeCapture([1])) as factory:
            capture = open_capture('rtsp://example/live', decoder_threads=0)
            capture.release()
        parameters = dict(zip(factory.call_args.args[2][::2], factory.call_args.args[2][1::2]))
        self.assertEqual(parameters[cv2.CAP_PROP_N_THREADS], 0)


if __name__ == '__main__':
    unittest.main()
