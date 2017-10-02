import numpy as np
import unittest

from thermography.utils.geometry import *


class TestGeometryUtils(unittest.TestCase):
    def setUp(self):
        pass

    def assertListAlmostEqual(self, first, second, places=None, msg=None):
        """
        Tests whether the elements of two lists are almost equal.
        :param first: The first list to compare.
        :param second: The second list to compare.
        :param places: Decimal places to be checked for comparison.
        :param msg: Optional error message in case comparison failure.
        :return: True if the two lists passed as argument are almost equal, False otherwise.
        """
        self.assertEqual(len(first), len(second),
                         msg="Compared lists are not of the same size. Give sizes: first = {}, second = {}".format(
                             len(first), len(second)))
        for f, s in zip(first, second):
            self.assertAlmostEqual(f, s, places=places, msg=msg)

    def test_segment_angle(self):
        """
        Tests the 'angle' function which computes the angle for a segment.
        """
        # Note that we test the angle based on pixel coordinates, i.e. with negated y coordinates.
        segment1 = [0, 0, 1, 0]
        self.assertAlmostEqual(angle(segment1[0:2], segment1[2:4]), 0.0 / 180 * np.pi)
        self.assertAlmostEqual(angle(segment1[2:4], segment1[0:2]), 0.0 / 180 * np.pi)

        segment2 = [0, 0, 1, 1]
        self.assertAlmostEqual(angle(segment2[0:2], segment2[2:4]), 135.0 / 180 * np.pi)
        self.assertAlmostEqual(angle(segment2[2:4], segment2[0:2]), 135.0 / 180 * np.pi)

        segment3 = [0, 0, 0, 1]
        self.assertAlmostEqual(angle(segment3[0:2], segment3[2:4]), 90.0 / 180 * np.pi)
        self.assertAlmostEqual(angle(segment3[2:4], segment3[0:2]), 90.0 / 180 * np.pi)

        segment4 = [0, 0, -1, 1]
        self.assertAlmostEqual(angle(segment4[0:2], segment4[2:4]), 45.0 / 180 * np.pi)
        self.assertAlmostEqual(angle(segment4[2:4], segment4[0:2]), 45.0 / 180 * np.pi)

        segment5 = [1.5, 1.5, 2.5, 2.5]
        self.assertAlmostEqual(angle(segment5[0:2], segment5[2:4]), 135.0 / 180 * np.pi)
        self.assertAlmostEqual(angle(segment5[2:4], segment5[0:2]), 135.0 / 180 * np.pi)

    def test_angle_difference(self):
        """
        Tests the 'angle_diff' function which computes the angle difference between two segments.
        """
        angle1 = 0.0
        self.assertAlmostEqual(angle_diff(angle1, angle1), 0.0)

        angle2 = np.pi * 0.5
        self.assertAlmostEqual(angle_diff(angle1, angle2), np.pi * 0.5)
        self.assertAlmostEqual(angle_diff(angle2, angle1), np.pi * 0.5)

        angle3 = -np.pi * 0.5
        self.assertAlmostEqual(angle_diff(angle1, angle3), np.pi * 0.5)
        self.assertAlmostEqual(angle_diff(angle3, angle1), np.pi * 0.5)

    def test_mean_segment_angle(self):
        """
        Tests the 'mean_segment_angle' function which computes the mean angle from a set of segments.
        """
        segment1 = [0, 0, 1, 0]
        segment2 = [0, -1, 1, -1]
        segment3 = [0, 0, 1, 1]
        segment4 = [0, 0, 1, -1]

        segments = np.array([segment1, segment2, segment3, segment4])
        self.assertAlmostEqual(mean_segment_angle(segments), 0.0)

    def test_segment_min_distance(self):
        """
        Tests the 'segment_min_distance' function which computes the minimal distance between two segments.
        """
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
        """
        Tests the 'line_estimate' function which computes a line estimate from two segments.
        """
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

    def test_merge_segments(self):
        """
        Tests the 'merge_segments' function which merges two segments into a single segment.
        """

        def sort_points(segment):
            """
            Sorts the endpoints of the segment passed as argument in lexicographical order.
            :param segment: Segment to be sorted.
            :return: Sorted segment.
            """
            pt1 = segment[0:2]
            pt2 = segment[2:4]
            if pt1[0] > pt2[0]:
                tmp = pt1
                pt1 = pt2
                pt2 = tmp
            elif pt1[0] == pt2[0]:
                if pt1[1] <= pt2[1]:
                    return segment
                else:
                    return np.array([*pt2, *pt1])

            return np.array([*pt1, *pt2])

        segment1 = np.array([0, 0, 1, 0])
        segment2 = np.array([2, 0, 3, 0])
        merged_segment = merge_segments([segment1, segment2])
        merged_segment = sort_points(merged_segment)

        self.assertListAlmostEqual(merged_segment, [0, 0, 3, 0])

        segment3 = np.array([0, 0.1, 1, 0.1])
        merged_segment = merge_segments([segment1, segment3])
        merged_segment = sort_points(merged_segment)

        self.assertListAlmostEqual(merged_segment, [0, 0.05, 1, 0.05])

        segment4 = np.array([0, 0, 0, 1])
        segment5 = np.array([0.1, 1.5, 0.1, 2.5])
        merged_segment = merge_segments([segment4, segment5])
        merged_segment = sort_points(merged_segment)

        self.assertListAlmostEqual(merged_segment, [0, 0.16666666, 0.1, 2.33333333])

    def test_point_line_distance(self):
        """
        Tests the 'point_line_distance' function which computes the distance between a point and a line.
        """
        slope = 1
        intercept = 0
        point1 = [0, 0]
        point2 = [1.5, 1.5]
        point3 = [-0.5, -0.5]
        self.assertAlmostEqual(point_line_distance(point1, slope, intercept), 0.0)
        self.assertAlmostEqual(point_line_distance(point2, slope, intercept), 0.0)
        self.assertAlmostEqual(point_line_distance(point3, slope, intercept), 0.0)

        point4 = [0, 1]
        self.assertAlmostEqual(point_line_distance(point4, slope, intercept), np.sqrt(2) * 0.5)

        slope = 0.5
        intercept = 1
        self.assertAlmostEqual(point_line_distance(point1, slope, intercept), 0.894427190)
        self.assertAlmostEqual(point_line_distance(point2, slope, intercept), 0.223606797)
        self.assertAlmostEqual(point_line_distance(point3, slope, intercept), 1.118033989)
        self.assertAlmostEqual(point_line_distance(point4, slope, intercept), 0.0)

    def test_segments_collinear(self):
        """
        Tests the 'segments_collinear' function which computes whether two segments are almost collinear or not.
        """
        segment1 = [0, 0, 1, 0]
        segment2 = [0.5, 0, 1.5, 0]
        self.assertTrue(segments_collinear(segment1, segment2, max_angle=0.05 / 180 * np.pi, max_endpoint_distance=0.1))

        segment3 = [0, 1, 1, 1]
        self.assertFalse(segments_collinear(segment1, segment3, max_angle=5 / 180 * np.pi, max_endpoint_distance=0.1))
        self.assertTrue(segments_collinear(segment1, segment3, max_angle=0.05 / 180 * np.pi, max_endpoint_distance=3))

        segment4 = [0.5, -1, 0.5, 1]
        self.assertFalse(segments_collinear(segment1, segment4, max_angle=10.0 / 180 * np.pi, max_endpoint_distance=5))
        self.assertTrue(segments_collinear(segment1, segment4, max_angle=np.pi, max_endpoint_distance=5))

    def test_segment_segment_intersection(self):
        """
        Tests the 'segment_segment_intersection' function which computes the intersection point between two segments.
        """
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
        """
        Tests the 'segment_line_intersection' function which computes the intersection point between a segment and a
        line.
        """
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

    def test_segment_sorting(self):
        """
        Tests the 'sort_segments' function which sorts a set of almost collinear segments based on their mean normal
        direction.
        """
        segments = np.array([[0, 0, 1, 0], [0, 1, 1, 1], [0, 1, 1, 1.1], [0, -1, 1, -0.5]])
        sorted_segments_indices = sort_segments(segments)
        self.assertTrue((sorted_segments_indices == [2, 1, 0, 3]).all())


if __name__ == '__main__':
    unittest.main()
