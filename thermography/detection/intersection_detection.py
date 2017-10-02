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

        self.raw_intersections = None

    def detect(self):
        """
        Detects the intersections between the segments passed to the constructor using the parameters passed to the
        constructor.
        """
        intersections = []
        num_clusters = len(self.segments)
        for cluster_index_i in range(num_clusters):
            cluster_i = self.segments[cluster_index_i]
            for cluster_index_j in range(cluster_index_i, num_clusters):
                cluster_j = self.segments[cluster_index_j]
                for segment_i in cluster_i:
                    for segment_j in cluster_j:
                        intersection = segment_segment_intersection(seg1=segment_i, seg2=segment_j)
                        if intersection is not False:
                            intersections.append(intersection)

        self.raw_intersections = np.array(intersections)
