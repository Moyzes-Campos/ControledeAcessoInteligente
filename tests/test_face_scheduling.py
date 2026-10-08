import unittest

import numpy as np

from access_control.faces import IdentityMemory
from access_control.pipeline import _limit_face_checks, _tracks_requiring_face_check
from access_control.types import TrackIdentity, TrackedPose


def track(track_id):
    return TrackedPose(track_id, 0, np.zeros(4, dtype=np.float32), .9, False)


class FaceSchedulingTests(unittest.TestCase):
    def test_new_person_can_be_checked_during_another_person_cooldown(self):
        selected = _limit_face_checks([track(1), track(2)], {1: 10.0}, 10.2, .5, 1)
        self.assertEqual([item.track_id for item in selected], [2])

    def test_oldest_attempt_is_prioritized_and_batch_is_bounded(self):
        selected = _limit_face_checks([track(1), track(2), track(3)],
                                     {1: 10.0, 2: 8.0, 3: 9.0}, 11.0, .5, 2)
        self.assertEqual([item.track_id for item in selected], [2, 3])

    def test_unknown_person_retries_after_cooldown(self):
        self.assertEqual(_limit_face_checks([track(1)], {1: 10.0}, 10.4, .5, 1), [])
        self.assertEqual(len(_limit_face_checks([track(1)], {1: 10.0}, 10.5, .5, 1)), 1)

    def test_unknown_track_is_checked(self):
        memory = IdentityMemory()
        self.assertEqual(
            [item.track_id for item in _tracks_requiring_face_check(
                [track(1)], memory, {}, 30, 0)],
            [1],
        )

    def test_known_track_is_skipped_when_recheck_is_disabled(self):
        memory = IdentityMemory()
        memory._items[1] = TrackIdentity(label="Moyzes", known=True)
        self.assertEqual(
            _tracks_requiring_face_check([track(1)], memory, {1: 9}, 30, 0),
            [],
        )

    def test_known_track_is_selected_after_configured_interval(self):
        memory = IdentityMemory()
        memory._items[1] = TrackIdentity(label="Moyzes", known=True)
        self.assertEqual(
            [item.track_id for item in _tracks_requiring_face_check(
                [track(1)], memory, {1: 10}, 30, 20)],
            [1],
        )
        self.assertEqual(
            _tracks_requiring_face_check([track(1)], memory, {1: 11}, 30, 20),
            [],
        )


if __name__ == "__main__":
    unittest.main()
