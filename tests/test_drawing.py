import unittest
from unittest.mock import patch

import numpy as np

from access_control.access import AccessState
from access_control.drawing import draw_access_overlay, draw_scene
from access_control.line_crossing import LineCounter
from access_control.types import FaceResult, PoseDetection, TrackIdentity, TrackedPose


class DrawingTests(unittest.TestCase):
    def test_preview_boxes_faces_and_pose_use_original_coordinates_without_mutation(self):
        frame = np.zeros((100, 200, 3), np.uint8)
        points = np.tile([100, 200, .9], (17, 1)).astype(np.float32)
        bbox = np.array([40, 80, 240, 320], dtype=np.float32)
        detection = PoseDetection(bbox, .9, points)
        track = TrackedPose(1, 0, bbox, .9, False)
        face = FaceResult((100, 100, 40, 80), 'Pessoa', .8, True)
        with patch('access_control.drawing._label'), \
             patch('access_control.drawing.cv2.rectangle') as rectangle, \
             patch('access_control.drawing.cv2.circle') as circle, \
             patch('access_control.drawing.cv2.line') as line:
            result = draw_scene(frame, [detection], [track], {1: TrackIdentity('Pessoa', .8, True)},
                                [face], 20, coordinate_scale=(.5, .25))
        self.assertEqual(rectangle.call_args_list[0].args[1:3], ((20, 20), (120, 80)))
        self.assertEqual(rectangle.call_args_list[1].args[1:3], ((50, 25), (70, 45)))
        self.assertTrue(all(call.args[1] == (50, 50) for call in circle.call_args_list))
        self.assertTrue(all(call.args[1:3] == ((50, 50), (50, 50)) for call in line.call_args_list))
        np.testing.assert_array_equal(bbox, [40, 80, 240, 320])
        np.testing.assert_allclose(points, np.tile([100, 200, .9], (17, 1)))
        self.assertFalse(np.shares_memory(result, frame))

    def test_preview_line_and_center_scale_without_changing_crossing_counter(self):
        frame = np.zeros((100, 200, 3), np.uint8)
        counter = LineCounter((0, 200), (400, 200), hysteresis=10)
        counter.update(1, (140, 150))
        counter.update(1, (140, 250))
        track = TrackedPose(1, 0, np.array([40, 80, 240, 320]), .9, False)
        state = AccessState('green', 0, 1, 0, 1, [])
        with patch('access_control.drawing._label'), \
             patch('access_control.drawing.cv2.line') as line, \
             patch('access_control.drawing.cv2.circle') as circle:
            draw_access_overlay(frame, [track], counter, state, None, coordinate_scale=(.5, .25))
        self.assertEqual(line.call_args.args[1:3], ((0, 50), (200, 50)))
        self.assertEqual(circle.call_args.args[1], (70, 50))
        self.assertEqual(counter.start, (0, 200))
        self.assertEqual(counter.end, (400, 200))
        self.assertEqual(counter.entries, 1)
        self.assertEqual(counter.exits, 0)
