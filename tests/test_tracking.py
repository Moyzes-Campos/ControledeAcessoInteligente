from __future__ import annotations

import unittest

import numpy as np

from access_control.tracking import OCSortPersonTracker
from access_control.types import PoseDetection


def detection(x1: float, y1: float, x2: float, y2: float) -> PoseDetection:
    return PoseDetection(
        bbox_xyxy=np.asarray([x1, y1, x2, y2], dtype=np.float32),
        confidence=0.9,
        keypoints_xyc=np.zeros((17, 3), dtype=np.float32),
    )


class OCSortTests(unittest.TestCase):
    def test_id_is_stable_while_person_moves(self) -> None:
        tracker = OCSortPersonTracker(iou_threshold=0.2)
        first = tracker.update([detection(10, 10, 60, 110)])
        second = tracker.update([detection(14, 10, 64, 110)])
        third = tracker.update([detection(20, 10, 70, 110)])
        self.assertEqual([first[0].track_id, second[0].track_id, third[0].track_id], [1, 1, 1])

    def test_new_person_gets_new_id(self) -> None:
        tracker = OCSortPersonTracker()
        result = tracker.update([detection(10, 10, 60, 110), detection(200, 10, 250, 110)])
        self.assertEqual([item.track_id for item in result], [1, 2])


if __name__ == "__main__":
    unittest.main()

