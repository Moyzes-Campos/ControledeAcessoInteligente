import math
import unittest

from access_control.timing import FrameRateLimiter, FrameRateMeter, processing_fps_limit


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class TimingTests(unittest.TestCase):
    def test_20ms_processing_with_20fps_camera_reports_20fps(self):
        clock = FakeClock()
        meter = FrameRateMeter(clock=clock)
        clock.now = .02
        self.assertEqual(meter.update(), 0.0)
        for index in range(1, 41):
            clock.now = index * .05 + .02
            self.assertAlmostEqual(meter.update(), 20.0)

    def test_slow_frame_reduces_measured_fps(self):
        clock = FakeClock()
        meter = FrameRateMeter(clock=clock)
        meter.update()
        clock.now = .05
        self.assertAlmostEqual(meter.update(), 20.0)
        clock.now = 2.05
        self.assertAlmostEqual(meter.update(), .5)

    def test_limiter_waits_only_remaining_time_and_does_not_catch_up(self):
        clock = FakeClock()
        limiter = FrameRateLimiter(20, clock=clock, sleep=clock.sleep)
        limiter.wait()
        self.assertEqual(clock.sleeps, [])
        clock.now = .02
        limiter.wait()
        self.assertAlmostEqual(clock.sleeps[-1], .03)
        clock.now = .2  # An overrun should not start a burst of overdue frames.
        limiter.wait()
        self.assertEqual(len(clock.sleeps), 1)
        clock.now = .22
        limiter.wait()
        self.assertAlmostEqual(clock.sleeps[-1], .03)

    def test_offline_limiter_never_sleeps(self):
        clock = FakeClock()
        limiter = FrameRateLimiter(processing_fps_limit(20, 10, live=False),
                                   clock=clock, sleep=clock.sleep)
        for _ in range(10):
            limiter.wait()
        self.assertEqual(clock.sleeps, [])

    def test_sleep_overshoot_does_not_accumulate_each_frame(self):
        clock = FakeClock()

        def oversleep(seconds):
            clock.sleep(seconds + .01)

        limiter = FrameRateLimiter(20, clock=clock, sleep=oversleep)
        limiter.wait()
        for index in range(1, 11):
            clock.now += .02
            limiter.wait()
            self.assertAlmostEqual(clock.now, index * .05 + .01)

    def test_limit_uses_camera_fps_and_allows_a_lower_explicit_value(self):
        self.assertEqual(processing_fps_limit(20, 0, live=True), 20)
        self.assertEqual(processing_fps_limit(20, 10, live=True), 10)
        self.assertEqual(processing_fps_limit(20, 50, live=True), 20)

    def test_invalid_camera_metadata_does_not_impose_a_fallback_rate(self):
        for source_fps in (0, -1, math.nan, math.inf):
            with self.subTest(source_fps=source_fps):
                self.assertEqual(processing_fps_limit(source_fps, 0, live=True), 0)
                self.assertEqual(processing_fps_limit(source_fps, 20, live=True), 20)

    def test_invalid_requested_limit_is_rejected(self):
        for requested in (-1, math.nan, math.inf):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                processing_fps_limit(20, requested, live=True)


if __name__ == "__main__":
    unittest.main()
