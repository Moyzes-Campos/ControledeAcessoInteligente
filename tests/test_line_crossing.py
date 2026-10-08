import unittest

from access_control.line_crossing import LineCounter, LineSettings, signed_distance


class LineCrossingTests(unittest.TestCase):
    def test_signed_distance_distinguishes_line_sides(self):
        self.assertLess(signed_distance((50, 20), (0, 50), (100, 50)), 0)
        self.assertGreater(signed_distance((50, 80), (0, 50), (100, 50)), 0)

    def test_crossing_counts_once_after_hysteresis(self):
        counter = LineCounter((0, 50), (100, 50), hysteresis=5)
        self.assertIsNone(counter.update(7, (50, 70)))
        self.assertIsNone(counter.update(7, (50, 52)))
        crossing = counter.update(7, (50, 40))
        self.assertEqual(crossing.direction, "saida")
        self.assertEqual((counter.entries, counter.exits), (0, 1))
        self.assertIsNone(counter.update(7, (50, 35)))

    def test_invert_swaps_direction(self):
        counter = LineCounter((0, 50), (100, 50), hysteresis=2, invert=True)
        counter.update(1, (50, 70))
        crossing = counter.update(1, (50, 40))
        self.assertEqual(crossing.direction, "entrada")

    def test_prune_forgets_departed_track(self):
        counter = LineCounter((0, 50), (100, 50))
        counter.update(3, (50, 80))
        counter.prune(set())
        self.assertIsNone(counter.update(3, (50, 20)))

    def test_crossing_outside_line_segment_is_ignored(self):
        counter = LineCounter((20, 50), (80, 50), hysteresis=2)
        self.assertIsNone(counter.update(9, (100, 70)))
        self.assertIsNone(counter.update(9, (100, 30)))
        self.assertEqual((counter.entries, counter.exits), (0, 0))

    def test_walking_around_endpoint_does_not_create_a_crossing(self):
        counter = LineCounter((20, 50), (80, 50), hysteresis=2)
        for point in ((50, 70), (100, 70), (100, 30), (50, 30)):
            self.assertIsNone(counter.update(9, point))
        self.assertEqual((counter.entries, counter.exits), (0, 0))

    def test_diagonal_motion_counts_only_if_it_intersects_segment(self):
        counter = LineCounter((20, 50), (80, 50), hysteresis=2)
        counter.update(9, (0, 30))
        crossing = counter.update(9, (100, 70))
        self.assertEqual(crossing.point, (50, 50))
        self.assertEqual(counter.entries, 1)

    def test_line_configuration_rejects_invalid_geometry(self):
        for values in ((0, 0, 0, 0), (0, 0, 2, 1), (0, 0, float('nan'), 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LineSettings(values)


if __name__ == "__main__":
    unittest.main()
