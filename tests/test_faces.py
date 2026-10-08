import unittest
from unittest.mock import Mock

import numpy as np

from access_control.faces import FaceRecognizer
from access_control.types import TrackedPose


class FaceCropTests(unittest.TestCase):
    def test_face_in_lower_half_is_included_and_coordinates_are_restored(self):
        subject = FaceRecognizer.__new__(FaceRecognizer)
        local_face = np.array([35, 170, 30, 35, 40, 175, 55, 175,
                               47, 185, 40, 195, 55, 195, .9], dtype=np.float32)
        subject._detect_raw = Mock(return_value=[local_face])
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        track = TrackedPose(1, 0, np.array([50, 40, 150, 240]), .9, False)
        faces = subject._detect_in_people(frame, [track])
        crop = subject._detect_raw.call_args.args[0]
        self.assertEqual(crop.shape, (220, 116, 3))
        self.assertEqual(faces[0][:4].tolist(), [77, 200, 30, 35])
        self.assertEqual(faces[0][4:6].tolist(), [82, 205])
        self.assertEqual(faces[0][-1], local_face[-1])

    def test_crop_is_clipped_at_frame_edges(self):
        subject = FaceRecognizer.__new__(FaceRecognizer)
        subject._detect_raw = Mock(return_value=[])
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        track = TrackedPose(1, 0, np.array([0, 0, 100, 200]), .9, False)
        self.assertEqual(subject._detect_in_people(frame, [track]), [])
        self.assertEqual(subject._detect_raw.call_args.args[0].shape, frame.shape)
