from thermography.utils.geometry import aspect_ratio, area
import numpy as np

__all__ = ["RectangleDetector", "RectangleDetectorParams"]

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
                    exit(0)

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
            print(segment_index_i, intersections_with_i)
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
