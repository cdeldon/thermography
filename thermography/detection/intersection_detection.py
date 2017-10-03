from thermography.utils.geometry import segment_segment_intersection
import numpy as np

__all__ = ["IntersectionDetector", "IntersectionDetectorParams"]


class IntersectionDetectorParams:
    def __init__(self):
        self.intersection_min_distance = 10


class IntersectionDetector:
    def __init__(self, input_segments: np.ndarray, params: IntersectionDetectorParams = IntersectionDetectorParams()):
        self.segments = input_segments
        self.params = params

        # Collection of intersections divided by clusters:
        # self.cluster_cluster_intersections[i,j] contains the intersections between cluster i and cluster j.
        # Note that i must be smaller-equal than j.
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
            cluster_i = self.segments[cluster_index_i]
            for cluster_index_j in range(cluster_index_i, num_clusters):
                cluster_j = self.segments[cluster_index_j]
                cluster_cluster_intersections = {}
                cluster_cluster_intersections_raw = []
                for i, segment_i in enumerate(cluster_i):
                    for j, segment_j in enumerate(cluster_j):
                        intersection = segment_segment_intersection(seg1=segment_i, seg2=segment_j)
                        if intersection is not False:
                            cluster_cluster_intersections[i,j] = intersection
                            cluster_cluster_intersections_raw.append(intersection)
                self.cluster_cluster_intersections[cluster_index_i,cluster_index_j] = cluster_cluster_intersections
                self.raw_intersections.extend(cluster_cluster_intersections_raw)

