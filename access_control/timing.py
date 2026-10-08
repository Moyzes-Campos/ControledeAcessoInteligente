from __future__ import annotations

from collections import deque
import math
import time


def processing_fps_limit(source_fps: float, requested: float, live: bool) -> float:
    if not math.isfinite(requested) or requested < 0:
        raise ValueError("O limite de FPS deve ser finito e maior ou igual a zero")
    if not live:
        return 0.0
    limits = [requested] if requested > 0 else []
    if math.isfinite(source_fps) and source_fps > 0:
        limits.append(source_fps)
    return min(limits) if limits else 0.0


class FrameRateLimiter:
    """Pace latest-frame reads; compensate sleep jitter and reset after long stalls."""

    def __init__(self, fps: float, clock=time.perf_counter, sleep=time.sleep):
        self.interval = 1.0 / fps if fps > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._next_frame: float | None = None

    def wait(self) -> None:
        if not self.interval:
            return
        now = self._clock()
        if self._next_frame is not None and now < self._next_frame:
            self._sleep(self._next_frame - now)
            now = self._clock()
        if self._next_frame is None or now >= self._next_frame + self.interval:
            self._next_frame = now + self.interval
        else:
            # sleep() can overshoot its deadline. Keep the original schedule so
            # small timer errors do not accumulate on every frame.
            self._next_frame += self.interval


class FrameRateMeter:
    """Measure delivered updates, including capture wait and the previous frame's I/O."""

    def __init__(self, window_seconds: float = 1.0, clock=time.perf_counter):
        self.window_seconds = window_seconds
        self._clock = clock
        self._samples: deque[float] = deque()

    def update(self) -> float:
        now = self._clock()
        self._samples.append(now)
        while len(self._samples) > 2 and self._samples[0] < now - self.window_seconds:
            self._samples.popleft()
        duration = now - self._samples[0]
        return (len(self._samples) - 1) / duration if duration > 0 else 0.0
