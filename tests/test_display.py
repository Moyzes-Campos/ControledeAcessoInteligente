import unittest
from unittest.mock import patch

import numpy as np

from access_control.display import PreviewRenderer, fit_window_size, map_display_point, poll_display_key


class DisplayTests(unittest.TestCase):
    def test_large_landscape_frame_fits_screen(self):
        self.assertEqual(fit_window_size(1920, 1080, (1366, 728)), (1164, 655))

    def test_large_portrait_frame_fits_screen(self):
        self.assertEqual(fit_window_size(1080, 1920, (1366, 728)), (368, 655))

    def test_small_frame_keeps_original_size(self):
        self.assertEqual(fit_window_size(640, 480, (1920, 1080)), (640, 480))

    def test_mouse_point_maps_from_preview_to_original_frame(self):
        self.assertEqual(
            map_display_point((582, 328), (1164, 655), (1920, 1080)),
            (960, 541),
        )

    def test_mouse_point_is_clamped_inside_original_frame(self):
        self.assertEqual(map_display_point((640, 480), (640, 480), (640, 480)), (639, 479))

    def test_preview_limits_large_camera_frame_without_modifying_original(self):
        frame = np.full((1520, 2688, 3), 127, dtype=np.uint8)
        preview = PreviewRenderer(1280, (3840, 2160)).prepare(frame)
        self.assertEqual(preview.shape, (724, 1280, 3))
        self.assertFalse(np.shares_memory(preview, frame))
        self.assertEqual(frame.shape, (1520, 2688, 3))
        self.assertTrue(np.all(frame == 127))

    def test_preview_recalculates_when_camera_orientation_changes(self):
        renderer = PreviewRenderer(1280, (1366, 728))
        landscape = renderer.prepare(np.zeros((1080, 1920, 3), np.uint8))
        portrait = renderer.prepare(np.zeros((1920, 1080, 3), np.uint8))
        self.assertEqual(landscape.shape[:2], (655, 1164))
        self.assertEqual(portrait.shape[:2], (655, 368))

    def test_disabled_preview_and_small_frames_keep_original_array(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertIs(PreviewRenderer(0).prepare(frame), frame)
        self.assertIs(PreviewRenderer(1280, (1920, 1080)).prepare(frame), frame)

    def test_display_key_uses_nonblocking_poll_and_preserves_quit_key(self):
        with patch('access_control.display.cv2.pollKey', return_value=ord('q')) as poll, \
             patch('access_control.display.cv2.waitKey') as wait:
            self.assertEqual(poll_display_key(), ord('q'))
        poll.assert_called_once_with()
        wait.assert_not_called()

    def test_display_key_falls_back_when_poll_is_unavailable(self):
        with patch('access_control.display.cv2.pollKey', None), \
             patch('access_control.display.cv2.waitKey', return_value=27) as wait:
            self.assertEqual(poll_display_key(), 27)
        wait.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
