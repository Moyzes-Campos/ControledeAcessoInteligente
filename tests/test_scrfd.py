import unittest

import numpy as np

from access_control.scrfd import _nms, _restore_detection


def detection(x, y, w, h, score=.9):
    landmarks = [x + 1, y + 1] * 5
    return np.asarray([x, y, w, h, *landmarks, score], dtype=np.float32)


class ScrfdAdapterTests(unittest.TestCase):
    def test_clockwise_coordinates_return_to_original_frame(self):
        row = detection(20, 10, 30, 40)
        restored = _restore_detection(row, 90, width=200, height=100)
        np.testing.assert_allclose(restored[:4], [10, 49, 40, 30])
        np.testing.assert_allclose(restored[4:6], [11, 78])

    def test_nms_keeps_best_overlapping_rotation_result(self):
        high = detection(10, 10, 50, 50, .9)
        low = detection(12, 12, 50, 50, .7)
        separate = detection(100, 100, 20, 20, .6)
        result = _nms([low, separate, high], .4)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(float(result[0][-1]), .9)


if __name__ == "__main__":
    unittest.main()
