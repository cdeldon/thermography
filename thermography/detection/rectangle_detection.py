from thermography.utils.geometry import aspect_ratio, angle, angle_diff, area, segment_segment_intersection
import numpy as np

__all__ = ["IntersectionDetector", "IntersectionDetectorParams",
           "RectangleDetector", "RectangleDetectorParams"]


class IntersectionDetectorParams:
    def __init__(self):
        # All intersections between segments whose relative angle is larger than this threshold are ignored.
        self.angle_threshold = np.pi / 180 * 10


class IntersectionDetector:
    def __init__(self, input_segments: np.ndarray, params: IntersectionDetectorParams = IntersectionDetectorParams()):
        self.segments = input_segments
        self.params = params

        # Collection of intersections divided by clusters:
        # self.cluster_cluster_intersections[i,j] contains the intersections between cluster i and cluster j.
        self.cluster_cluster_intersections = {}
        self.raw_intersections = []

    def detect(self):
        """
        Detects the intersections between the segments passed to the constructor using the parameters passed to the
        constructor.
        """
        self.cluster_cluster_intersections = {}
        self.raw_intersections = []
        num_clusters = len(self.segments)
        for cluster_index_i in range(num_clusters):
            for cluster_index_j in range(num_clusters):
                self.__detect_intersections_between_clusters(cluster_index_i, cluster_index_j)

    def __detect_intersections_between_clusters(self, cluster_index_i: int, cluster_index_j: int):
        cluster_i = self.segments[cluster_index_i]
        cluster_j = self.segments[cluster_index_j]
        self.cluster_cluster_intersections[cluster_index_i, cluster_index_j] = {}
        cluster_cluster_intersections_raw = []
        for i, segment_i in enumerate(cluster_i):
            intersections_with_i = {}
            angle_i = angle(segment_i[0:2], segment_i[2:4])
            for j, segment_j in enumerate(cluster_j):
                angle_j = angle(segment_j[0:2], segment_j[2:4])
                d_angle = angle_diff(angle_i, angle_j)
                if np.pi / 2 - d_angle > self.params.angle_threshold:
                    continue
                intersection = segment_segment_intersection(seg1=segment_i, seg2=segment_j)
                if intersection is not False:
                    intersections_with_i[j] = intersection
                    cluster_cluster_intersections_raw.append(intersection)
            self.cluster_cluster_intersections[cluster_index_i, cluster_index_j][i] = intersections_with_i
        if cluster_index_j >= cluster_index_i:
            self.raw_intersections.extend(cluster_cluster_intersections_raw)


class RectangleDetectorParams:
    def __init__(self):
        self.aspect_ratio = 1.0
        self.aspect_ratio_relative_deviation = 0.5

        self.min_area = 20 * 40


class RectangleDetector:
    def __init__(self, input_intersections: dict, params: RectangleDetectorParams = RectangleDetectorParams()):
        self.intersections = input_intersections
        self.params = params

        self.rectangles = []

    def detect(self):
        # Iterate over each pair of clusters.
        num_clusters = int((np.sqrt(8 * len(self.intersections) + 1) - 1) / 2)
        for cluster_index_i in range(num_clusters):
            for cluster_index_j in range(cluster_index_i + 1, num_clusters):
                if (cluster_index_i, cluster_index_j) in self.intersections:
                    self.__detect_rectangles_between_clusters(cluster_index_i, cluster_index_j)

    @staticmethod
    def fulfills_ratio(rectangle: np.ndarray, expected_ratio: float, deviation: float) -> bool:
        ratio = aspect_ratio(rectangle)

        if np.abs(expected_ratio - ratio) / expected_ratio < deviation:
            return True
        if np.abs(expected_ratio - 1.0 / ratio) / expected_ratio < deviation:
            return True
        return False

    def __detect_rectangles_between_clusters(self, cluster_index_i: int, cluster_index_j: int):
        intersections_i_j = self.intersections[cluster_index_i, cluster_index_j]
        rectangles = []
        for segment_index_i, intersections_with_i in intersections_i_j.items():
            if segment_index_i + 1 not in intersections_i_j:
                continue

            intersections_with_i_plus = intersections_i_j[segment_index_i + 1]
            for segment_index_j, intersection in intersections_with_i.items():
                if segment_index_j + 1 not in intersections_with_i:
                    continue
                if segment_index_j in intersections_with_i_plus and segment_index_j + 1 in intersections_with_i_plus:
                    coord1 = intersections_with_i[segment_index_j]
                    coord2 = intersections_with_i[segment_index_j + 1]
                    coord3 = intersections_with_i_plus[segment_index_j]
                    coord4 = intersections_with_i_plus[segment_index_j + 1]
                    rectangle = np.array([coord1, coord2, coord4, coord3])
                    if self.fulfills_ratio(rectangle, self.params.aspect_ratio,
                                           self.params.aspect_ratio_relative_deviation) and area(rectangle) >= self.params.min_area:
                        rectangles.append(rectangle)

        self.rectangles.extend(rectangles)
