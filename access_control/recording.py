from __future__ import annotations

import math
import time


class RealtimeVideoWriter:
    """Keep the displayed frame until its next update on a fixed FPS timeline."""

    def __init__(self, writer, fps: float, clock=time.perf_counter):
        self.writer = writer
        self.fps = fps
        self.clock = clock
        self.origin = None
        self.previous = None
        self.frames_written = 0

    def _fill_until(self, timestamp):
        target = math.ceil(max(0.0, timestamp - self.origin) * self.fps)
        while self.frames_written < target:
            self.writer.write(self.previous)
            self.frames_written += 1

    def write(self, frame):
        now = self.clock()
        if self.origin is None:
            self.origin = now
        else:
            self._fill_until(now)
        self.previous = frame

    def release(self):
        try:
            if self.previous is not None:
                self._fill_until(self.clock())
                if not self.frames_written:
                    self.writer.write(self.previous)
                    self.frames_written = 1
        finally:
            self.writer.release()
