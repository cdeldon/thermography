import numpy as np
import unittest

from thermography.utils.geometry import *


class TestGeometryUtils(unittest.TestCase):
    def setUp(self):
        pass

    def assertListAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        self.assertEqual(len(first), len(second),
                         msg="Compared lists are not of the same size. Give sizes: first = {}, second = {}".format(
                             len(first), len(second)))
        for f, s in zip(first, second):
            self.assertAlmostEqual(f, s, places=places, msg=msg, delta=delta)

    def test_segment_angle(self):
        segment1 = [0, 0, 1, 0]
        self.assertAlmostEqual(angle(segment1[0:2], segment1[2:4]), 0.0 / 180 * np.pi)
        self.assertAlmostEqual(angle(segment1[2:4], segment1[0:2]), 0.0 / 180 * np.pi)

        segment2 = [0, 0, 1, 1]
        self.assertAlmostEqual(angle(segment2[0:2], segment2[2:4]), 45.0 / 180 * np.pi)
        self.assertAlmostEqual(angle(segment2[2:4], segment2[0:2]), 45.0 / 180 * np.pi)

        segment3 = [0, 0, 0, 1]
        self.assertAlmostEqual(angle(segment3[0:2], segment3[2:4]), 90.0 / 180 * np.pi)
        self.assertAlmostEqual(angle(segment3[2:4], segment3[0:2]), 90.0 / 180 * np.pi)

        segment4 = [0, 0, -1, 1]
        self.assertAlmostEqual(angle(segment4[0:2], segment4[2:4]), 135.0 / 180 * np.pi)
        self.assertAlmostEqual(angle(segment4[2:4], segment4[0:2]), 135.0 / 180 * np.pi)

        segment5 = [1.5, 1.5, 2.5, 2.5]
        self.assertAlmostEqual(angle(segment5[0:2], segment5[2:4]), 45.0 / 180 * np.pi)
        self.assertAlmostEqual(angle(segment5[2:4], segment5[0:2]), 45.0 / 180 * np.pi)

    def test_segment_min_distance(self):
        segment1 = np.array([0, 0, 1, 0])
        segment2 = np.array([0, 1, 1, 1])
        self.assertAlmostEqual(segment_min_distance(segment1, segment1), 0.0)
        self.assertAlmostEqual(segment_min_distance(segment1, segment2), 1.0)

        segment3 = np.array([0, 2, 0, 1])
        self.assertAlmostEqual(segment_min_distance(segment1, segment3), 1.0)

        segment4 = np.array([0.5, 0.5, 0.5, -0.5])
        self.assertAlmostEqual(segment_min_distance(segment1, segment4), 0.0)

        segment5 = np.array([1, 0, 0, 0])
        self.assertAlmostEqual(segment_min_distance(segment1, segment5), 0.0)

        segment6 = np.array([0.5, 1, 2, 1.5])
        self.assertAlmostEqual(segment_min_distance(segment1, segment6), 1)

    def test_line_estimate(self):
        segment1 = np.array([0, 0, 1, 0])
        segment2 = np.array([1, 0, 2, 0])
        self.assertListAlmostEqual(line_estimate(segment1, segment2), (0, 0))
        self.assertListAlmostEqual(line_estimate(segment2, segment1), (0, 0))

        segment3 = np.array([0, 0, 1, 0.001])
        self.assertListAlmostEqual(line_estimate(segment1, segment3), (0, 0), places=3)
        self.assertListAlmostEqual(line_estimate(segment3, segment1), (0, 0), places=3)

        segment4 = np.array([0, 0, 1, 1])
        segment5 = np.array([0.5, 0.5001, 1.5, 1.4999])
        self.assertListAlmostEqual(line_estimate(segment4, segment5), (1, 0), places=3)
        self.assertListAlmostEqual(line_estimate(segment5, segment4), (1, 0), places=3)

    def test_segment_segment_intersection(self):
        segment1 = np.array([0, 0, 1, 0])
        segment2 = np.array([0, 0, 0, 1])
        self.assertListAlmostEqual(segment_segment_intersection(segment1, segment2), [0, 0])
        self.assertListAlmostEqual(segment_segment_intersection(segment2, segment1), (0, 0))

        segment3 = np.array([0.5, 1, 0.5, -1])
        self.assertListAlmostEqual(segment_segment_intersection(segment1, segment3), [0.5, 0])
        self.assertListAlmostEqual(segment_segment_intersection(segment3, segment1), [0.5, 0])

        segment4 = np.array([0.3, 1, 0.7, -1])
        self.assertListAlmostEqual(segment_segment_intersection(segment1, segment4), [0.5, 0])
        self.assertListAlmostEqual(segment_segment_intersection(segment4, segment1), [0.5, 0])

        segment5 = np.array([0, 1, 1, 1])
        self.assertFalse(segment_segment_intersection(segment1, segment5))
        self.assertFalse(segment_segment_intersection(segment5, segment1))

        segment6 = np.array([1.5, 0, 2.5, 0])
        self.assertFalse(segment_segment_intersection(segment1, segment6))
        self.assertFalse(segment_segment_intersection(segment6, segment1))

    def test_segment_line_intersection(self):
        segment1 = np.array([0, 1, 1, 0])
        segment2 = np.array([1, 0, 0, 1])
        line1 = [1, 0]
        self.assertListAlmostEqual(segment_line_intersection(segment1, line1[0], line1[1]), [0.5, 0.5])
        self.assertListAlmostEqual(segment_line_intersection(segment2, line1[0], line1[1]), [0.5, 0.5])

        line2 = [1, 1.5]
        self.assertFalse(segment_line_intersection(segment1, line2[0], line2[1]))
        self.assertFalse(segment_line_intersection(segment2, line2[0], line2[1]))

        line3 = [-1, 0]
        self.assertFalse(segment_line_intersection(segment1, line3[0], line3[1]))
        self.assertFalse(segment_line_intersection(segment2, line3[0], line3[1]))


if __name__ == '__main__':
    unittest.main()
