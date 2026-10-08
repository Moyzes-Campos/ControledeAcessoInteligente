from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable

import cv2

from .timing import FrameRateMeter


def is_live_source(source: str) -> bool:
    return source.strip().isdigit() or "://" in source


def open_capture(source: str, decoder_threads: int = 1):
    source = source.strip()
    if is_live_source(source):
        return LatestFrameCapture(source, decoder_threads)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Nao foi possivel abrir o arquivo de video")
    return capture


class LatestFrameCapture:
    """Decode continuously, keeping only the latest frame, as in the PPE project.

    The reader owns VideoCapture, including reconnects and release, so shutdown
    never releases the decoder concurrently with a native read call.
    """

    def __init__(self, source: str, decoder_threads: int = 1):
        self.source = source
        if decoder_threads < 0:
            raise ValueError('O numero de threads de decodificacao nao pode ser negativo')
        self.decoder_threads = decoder_threads
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest = None
        self._sequence = 0
        self._consumed = 0
        self._properties = {}
        self._received_at = 0.0
        self._arrival_meter = FrameRateMeter()
        self._arrival_fps = 0.0
        self._delivered_frames = 0
        self._dropped_frames = 0
        self._latest_frame_info: dict = {}
        self._delivered_frame_info: dict = {}
        self._read_stalls_by_frame_type: dict[str, int] = {}
        self._thread = threading.Thread(target=self._reader_loop, name="camera-reader", daemon=True)
        self._thread.start()
        with self._condition:
            ready = self._condition.wait_for(lambda: self._latest is not None, timeout=15)
        if not ready:
            self.release()
            raise RuntimeError("Nao foi possivel receber quadros da camera em 15 segundos")

    def _connect(self):
        if self.source.lower().startswith(("rtsp://", "rtsps://")):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        if self.source.isdigit():
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            capture = cv2.VideoCapture(int(self.source), backend)
        else:
            parameters = [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
                # FFmpeg frame threading can release several frames together and
                # increase latency. One decoder thread delivered steadier live frames.
                cv2.CAP_PROP_N_THREADS, self.decoder_threads,
            ]
            capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, parameters)
        if not capture.isOpened():
            capture.release()
            return None
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _reader_loop(self):
        while not self._stop.is_set():
            try:
                capture = self._connect()
            except cv2.error:
                capture = None
            if capture is None:
                self._stop.wait(1)
                continue
            try:
                properties = {key: capture.get(key) for key in (
                    cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                    cv2.CAP_PROP_N_THREADS,
                )}
                while not self._stop.is_set():
                    read_started = time.perf_counter()
                    ok, frame = capture.read()
                    received_at = time.perf_counter()
                    if not ok or frame is None:
                        break
                    native_read_ms = (received_at - read_started) * 1000
                    frame_type = "unknown"
                    if not self.source.isdigit():
                        property_id = getattr(cv2, "CAP_PROP_FRAME_TYPE", None)
                        if property_id is not None:
                            type_code = capture.get(property_id)
                            frame_type = {73: "I", 80: "P", 66: "B"}.get(type_code, "unknown")
                    with self._condition:
                        self._properties = properties
                        self._latest = frame
                        self._sequence += 1
                        self._latest_frame_info = {
                            "sequence": self._sequence,
                            "frame_type": frame_type,
                            "native_read_ms": round(native_read_ms, 3),
                            "arrival_interval_ms": (round((received_at - self._received_at) * 1000, 3)
                                                    if self._received_at else None),
                        }
                        if native_read_ms >= 200:
                            self._read_stalls_by_frame_type[frame_type] = (
                                self._read_stalls_by_frame_type.get(frame_type, 0) + 1)
                        self._received_at = received_at
                        self._arrival_fps = self._arrival_meter.update()
                        self._condition.notify_all()
            except cv2.error:
                pass
            finally:
                capture.release()
            if not self._stop.is_set():
                print("Stream interrompido; reconectando a camera...", flush=True)
                self._stop.wait(1)

    def read(self, on_wait: Callable[[], bool] | None = None):
        """Wait for a new frame; an optional GUI callback can interrupt the wait.

        Never return the already consumed frame. The callback runs outside the
        capture lock so GUI event handling cannot prevent the reader publishing.
        """
        while True:
            with self._condition:
                ready = self._condition.wait_for(
                    lambda: self._sequence != self._consumed or self._stop.is_set(),
                    timeout=0.02 if on_wait is not None else 0.1,
                )
                if self._stop.is_set():
                    return False, None
                if ready:
                    self._dropped_frames += max(0, self._sequence - self._consumed - 1)
                    self._consumed = self._sequence
                    self._delivered_frames += 1
                    self._delivered_frame_info = dict(self._latest_frame_info)
                    return True, self._latest
            if on_wait is not None and not on_wait():
                return False, None

    def get_stats(self) -> dict:
        with self._condition:
            age = time.perf_counter() - self._received_at if self._received_at else None
            return {
                "received_fps": round(self._arrival_fps, 3) if age is not None and age < 1.0 else 0.0,
                "frames_received": self._sequence,
                "frames_delivered": self._delivered_frames,
                "frames_dropped": self._dropped_frames,
                "latest_frame_age_ms": round(age * 1000, 3) if age is not None else None,
                "decoder_threads": int(self._properties.get(cv2.CAP_PROP_N_THREADS, 0)),
                "delivered_frame": dict(self._delivered_frame_info),
                "native_read_stalls_ge_200ms": dict(self._read_stalls_by_frame_type),
            }

    def get(self, key):
        with self._condition:
            return self._properties.get(key, 0.0)

    def release(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=6)
