from thermography.utils.geometry import angle, angle_diff, segment_segment_intersection
import numpy as np

__all__ = ["IntersectionDetector", "IntersectionDetectorParams"]


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
