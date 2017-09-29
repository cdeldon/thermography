from thermography.utils.geometry import segment_segment_intersection
import numpy as np

__all__ = ["IntersectionDetector", "IntersectionDetectorParams"]

class IntersectionDetectorParams:
    def __init__(self):
        self.intersection_min_distance = 10


class IntersectionDetector:
    def __init__(self, input_segments: np.ndarray, params=IntersectionDetectorParams()):
        self.segments = input_segments
        self.params = params

        self.raw_intersections = None

    def detect(self):
        intersections = []
        num_segments = len(self.segments)
        for i in range(num_segments):
            segment_i = self.segments[i]
            for j in range(i + 1, num_segments):
                segment_j = self.segments[j]
                intersection = segment_segment_intersection(seg1=segment_i, seg2=segment_j)
                if intersection:
                    intersections.append(intersection)

        self.raw_intersections = np.array(intersections)
