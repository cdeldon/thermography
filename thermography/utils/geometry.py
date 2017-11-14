"""The functions implemented in this module are used bz :mod:`thermography` for the computation of geometric properties."""

import cv2
import numpy as np

__all__ = ["angle",
           "angle_diff",
           "aspect_ratio",
           "area",
           "area_between_rectangles",
           "line_estimate",
           "mean_segment_angle",
           "merge_segments",
           "point_line_distance",
           "rectangle_contains",
           "segments_collinear",
           "segment_line_intersection",
           "segment_min_distance",
           "segment_segment_intersection",
           "sort_rectangle",
           "sort_segments"]


def angle(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """Computes the angle (in radiants) between a segment specified by the two points and the x-axis.

    .. note:: The computed angle lies in [0, pi].

    :param pt1: First point of the segment.
    :param pt2: Second point of the segment.
    :return: Angle in radiants between the segment and the x-axis.
    """
    diff = pt2 - pt1
    a = np.arctan2(diff[1], diff[0])
    if np.abs(a % np.pi) <= 0.00001:
        return 0
    elif a < 0:
        a += np.pi
    return a


def angle_diff(angle1: float, angle2: float) -> float:
    """Computes the angle difference between the input arguments.

    .. note:: The resulting angle difference is in [0, pi * 0.5]

    :param angle1: First angle expressed in radiants.
    :param angle2: Second angle expressed in radiants.
    :return: Angle difference between the input parameters. This angle represents the smallest positive angle between the input parameters.
    """
    d_angle = np.abs(angle1 - angle2)
    d_angle = d_angle % np.pi
    if d_angle > np.pi * 0.5:
        d_angle -= np.pi
    return np.abs(d_angle)


def area(points: np.ndarray) -> float:
    """Computes the surface of the polygon defined by the coordinates passed as argument.

    :param points: List of coordinates defining  the polygon's vertices.
    :return: The surface contained by the polygon.
    """
    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def area_between_rectangles(rect1: np.ndarray, rect2: np.ndarray) -> float:
    """Computes the cumulative surface between the corresponding edges of the two rectangles passed as argument.

    ::

       *--------------------*
       |####################|
       |###*-------------*##|
       |###|             |##|
       |###|             |##|
       |###|             |##|
       |###*-------------*##|
       |####################|
       *--------------------*

    :param rect1: First rectangle's coordinates [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
    :param rect2: Second rectangle's coordinates [[x'0,y'0],[x'1,y'1],[x'2,y'2],[x'3,y'3]]
    :return: The surface between the rectangles' corresponding edges.
    """

    r0 = sort_rectangle(np.array([*rect1[0:2], *rect2[1::-1]]))
    r1 = sort_rectangle(np.array([*rect1[1:3], *rect2[2:0:-1]]))
    r2 = sort_rectangle(np.array([*rect1[2:4], *rect2[3:1:-1]]))
    r3 = sort_rectangle(np.array([rect1[3], rect1[0], rect2[0], rect2[3]]))

    a0, a1, a2, a3 = area(r0), area(r1), area(r2), area(r3)
    return a0 + a1 + a2 + a3


def aspect_ratio(rectangle: np.ndarray) -> float:
    """Computes the aspect ratio of a rectangle.

    The aspect ratio is computed based on the following rectangle.

    ::

         3      s2     2
         *-------------*
         |             |
      s4 |             | s3
         |             |
         *-------------*
         0     s1      1

    :param rectangle: Rectangle is a numpy array of coordinates ordered as shown in the diagram.
    :return: Aspect ratio of the rectangle.
    """
    s1 = rectangle[1] - rectangle[0]
    s2 = rectangle[2] - rectangle[3]
    s3 = rectangle[2] - rectangle[1]
    s4 = rectangle[3] - rectangle[0]
    dx = np.mean([np.linalg.norm(s1), np.linalg.norm(s2)])
    dy = np.mean([np.linalg.norm(s3), np.linalg.norm(s4)])
    return dx / dy


def line_estimate(seg1: np.ndarray, seg2: np.ndarray) -> tuple:
    """Computes the line estimation (regression) using the endpoints of the segments passed as argument.

    .. note:: Depending on the points' distribution, the line estimate is computed in the x-y plane as usual, or on the y-x plane (i.e. with inverted coordinates).

    :param seg1: First segment.
    :param seg2: Second segment.
    :return: The slope and intercept of the estimated line, a boolean indicating whether the slope and intercept refer to a vertical or horizontal polyfit.
    """
    x = [*seg1[0::2], *seg2[0::2]]
    y = [*seg1[1::2], *seg2[1::2]]

    # Compute vertical polyfit if standard deviation of y coordinates is larger than the one of x coordinates.
    std_x = np.std(x)
    std_y = np.std(y)

    vertical = std_y > std_x

    if vertical:
        [slope, intercept] = np.polyfit(y, x, 1)
    else:
        [slope, intercept] = np.polyfit(x, y, 1)

    return (slope, intercept), vertical


def mean_segment_angle(segment_list: list) -> float:
    """Computes the mean angle of a list of segments.

    .. note:: The computed mean angle lies in [0, pi]

    :param segment_list: A list of segments of the form [np.array([x0, y0, x1, y1]), np.array([...]), .... ]
    :return: The mean angle of the segments passed as argument.
    """
    complex_coordinates = []
    for segment in segment_list:
        a = angle(segment[0:2], segment[2:4]) * 2

        # Compute complex representation of angle.
        complex_coordinate = np.array([np.cos(a), np.sin(a)])
        complex_coordinates.append(complex_coordinate)

    mean_coordinate = np.mean(complex_coordinates, axis=0)
    a = np.arctan2(mean_coordinate[1], mean_coordinate[0]) / 2
    if a < 0:
        a += np.pi
    return a


def merge_segments(segment_list: list) -> np.ndarray:
    """Computes a unique segments as a representation of the almost collinear segments passed as argument.

    :param segment_list: List of almost collinear segments to be merged into a single segment.
    :return: A new segment defined on the line estimation over the segments passed as argument.
    """

    if len(segment_list) == 1:
        return segment_list[0]

    x = []
    y = []
    for segment in segment_list:
        x.append(segment[0])
        x.append(segment[2])
        y.append(segment[1])
        y.append(segment[3])

    std_x = np.std(x)
    std_y = np.std(y)

    vertical = std_y > std_x

    if vertical:
        [slope, intercept] = np.polyfit(y, x, 1)
        y0 = np.min(y)
        y1 = np.max(y)
        x0 = intercept + slope * y0
        x1 = intercept + slope * y1
    else:
        [slope, intercept] = np.polyfit(x, y, 1)
        x0 = np.min(x)
        x1 = np.max(x)
        y0 = intercept + slope * x0
        y1 = intercept + slope * x1

    return np.array([x0, y0, x1, y1])


def point_line_distance(point: np.ndarray, slope: float, intercept: float, vertical: bool) -> float:
    """Computes the shortest distance between a point and a line defined by its slope and intercept.

    :param point: Point given by a 2D coordinate in the form of [x, y]
    :param slope: Slope of the line
    :param intercept: Intercept of the line
    :param vertical: Boolean indicating if the slope and intercept refer to a vertical or horizontal polyfit.
    :return: Positive minimal distance between the point passed as argument and the line defined by the slope and intercept passed as arguments.
    """
    if vertical:
        point = [point[1], point[0]]
    return np.abs(-slope * point[0] + point[1] - intercept) / np.sqrt(1 + slope * slope)


def rectangle_contains(rectangle: np.ndarray, point: np.ndarray) -> bool:
    """Computes whether a point is inside a rectangle or not.

    :param rectangle: Rectangle to be tested against the query point.
    :param point: Point to be tested against the rectangle.
    :return: True if the point is inside or on the rectangle contours, False otherwise.
    """
    point = (int(point[0]), int(point[1]))
    rectangle = np.array([(int(r[0]), int(r[1])) for r in rectangle])
    return cv2.pointPolygonTest(rectangle, point, False) >= 0


def segments_collinear(seg1: np.ndarray, seg2: np.ndarray, max_angle: float = 5.0 / 180 * np.pi,
                       max_endpoint_distance: float = 50.0) -> bool:
    """Tests whether two segments are collinear given some thresholds for collinearity.

    :param seg1: First segment to be tested.
    :param seg2: Second segment to be tested.
    :param max_angle: Maximal angle between segments to be accepted as collinear.
    :param max_endpoint_distance: Max sum of euclidean distance between the endpoints of the passed segments and the line estimate computed between the segments. This parameter discards almost parallel segments with different intercept.
    :return: True if the segments are almost collinear, False otherwise.
    """
    # Compute the angle between the segments.
    a = angle_diff(angle(seg1[0:2], seg1[2:4]), angle(seg2[0:2], seg2[2:4]))
    if a > max_angle:
        return False

    intersection = segment_segment_intersection(np.array(seg1), np.array(seg2))
    if intersection is not False:
        return True
    else:
        (slope, intercept), vertical = line_estimate(seg1, seg2)
        dist_sum_2 = 0
        for point in [seg1[0:2], seg1[2:4], seg2[0:2], seg2[2:4]]:
            dist_sum_2 += point_line_distance(point, slope, intercept, vertical) ** 2
        if dist_sum_2 >= max_endpoint_distance ** 2:
            return False
        return True


def segment_line_intersection(seg: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Computes the intersection point between a segment and a line.

    .. note:: If no intersection is found, a boolean flag set to `False` is returned instead.

    :param seg: Segment to be intersected with the line.
    :param slope: Slope of the intersecting line.
    :param intercept: Intercept of the intersecting line.
    :return: Coordinates of the intersection point, or False if no intersection point is found.
    """
    pt1 = seg[0:2]
    pt2 = seg[2:4]

    x0 = np.array([0, intercept])
    x1 = np.array([1, slope + intercept])
    r = x1 - x0
    r = r / np.linalg.norm(r)

    d1 = pt1 - x0
    d2 = pt2 - x0

    if np.cross(d1, r).dot(np.cross(r, d2)) < 0:
        return False

    d1_proj = x0 + r * np.dot(d1, r)
    d2_proj = x0 + r * np.dot(d2, r)

    # If segment was perpendicular to line.
    if (d1_proj == d2_proj).all():
        return d1_proj

    return segment_segment_intersection(seg, np.array([d1_proj[0], d1_proj[1], d2_proj[0], d2_proj[1]]))


def segment_min_distance(seg1: np.ndarray, seg2: np.ndarray) -> float:
    """Computes the minimal distance between two segments.

    Implementation taken form `here <https://ch.mathworks.com/matlabcentral/fileexchange/32487-shortest-distance-between-two-line-segments?focused=3821416&tab=function>`_.

    :param seg1: First segment defined as [x1, y1, x2, y2]
    :param seg2: Second segment defined as [x2, y3, x4, y4]
    :return: The minimal distance between the two segments
    """
    p1 = seg1[0:2]
    p2 = seg1[2:4]
    p3 = seg2[0:2]
    p4 = seg2[2:4]

    u = (p1 - p2).astype(np.float64)
    v = (p3 - p4).astype(np.float64)
    w = (p2 - p4).astype(np.float64)

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a * c - b * b
    sD = D
    tD = D

    eps = 0.0001

    if D < eps:  # the lines are almost parallel
        sN = 0.0  # force using point P0 on segment S1
        sD = 1.0  # to prevent possible division by 0.0 later
        tN = e
        tD = c
    else:  # get the closest points on the infinite lines
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:  # sc < 0 = > the s=0 edge is visible
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:  # sc > 1 = > the s = 1 edge is visible
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:  # tc < 0 = > the t=0 edge is visible
        tN = 0.0
        # recompute sc for this edge
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:  # tc > 1 = > the t = 1 edge is visible
        tN = tD
        # recompute sc for this edge
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    # finally do the division to get sc and tc
    if np.abs(sN) < eps:
        sc = 0.0
    else:
        sc = sN / sD

    if np.abs(tN) < eps:
        tc = 0.0
    else:
        tc = tN / tD

    # get the difference of the two closest points
    dP = w + (sc * u) - (tc * v)  # = S1(sc) - S2(tc)
    distance = np.linalg.norm(dP)

    return distance


def segment_segment_intersection(seg1: np.ndarray, seg2: np.ndarray) -> np.ndarray:
    """Computes the intersection point between two segments.

    .. note:: If no intersection is found, a boolean flag set to `False` is returned.

    :param seg1: First segment of intersection.
    :param seg2: Second segment of intersection.
    :return: Coordinates of intersection point, or False if no intersection is found.
    """
    if (seg1 == seg2).all():
        return False
    s1_x = seg1[2] - seg1[0]
    s1_y = seg1[3] - seg1[1]
    s2_x = seg2[2] - seg2[0]
    s2_y = seg2[3] - seg2[1]

    denom = -s2_x * s1_y + s1_x * s2_y
    if denom == 0:
        return False

    s = (-s1_y * (seg1[0] - seg2[0]) + s1_x * (seg1[1] - seg2[1])) / denom
    t = (s2_x * (seg1[1] - seg2[1]) - s2_y * (seg1[0] - seg2[0])) / denom

    if 0 <= s <= 1 and 0 <= t <= 1:
        x = seg1[0] + (t * s1_x)
        y = seg1[1] + (t * s1_y)
        return np.array([x, y])
    return False


def sort_rectangle(rectangle: np.ndarray) -> np.ndarray:
    """Sorts the coordinates in the rectangle such that the final indexing corresponds to the following structure:

    ::

       +-----------> x
       |  3             2
       |  *-------------*
       |  |             |
       v  |             |
       y  |             |
          *-------------*
          0             1

    :param rectangle: numpy array of coordinates with form: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    :return: A rectangle whose vertices are sorted.
    """

    center = np.mean(rectangle, axis=0)
    diff = rectangle - center
    angles = np.arctan2(diff[:, 1], diff[:, 0])

    order = np.argsort(angles)
    return rectangle[order]


def sort_segments(segment_list: list) -> np.ndarray:
    """Sorts the segments passed as argument based on the normal associated to the mean angle.

    :param segment_list:  A list of segments of the form [[x0, y0, x1, y1], [...], .... ]
    :return: A list of indices in the segment list passed as argument which sorts the segments.
    """
    # Compute the mean angle of the segments in the list.
    mean_angle = mean_segment_angle(segment_list)

    # Compute the associated segment centers.
    segment_centers = np.array([(s[0:2] + s[2:4]) * 0.5 for s in segment_list])

    # Compute the normal associated to the mean angle.
    direction = np.array([np.cos(mean_angle), np.sin(mean_angle)])
    normal = np.array([-direction[1], direction[0]])

    # Project the segment centers along the normal defined by the mean angle.
    projected_centers = np.array([np.dot(center, normal) for center in segment_centers])
    order = np.argsort(projected_centers)

    return order
