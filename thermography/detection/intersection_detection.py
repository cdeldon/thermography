import numpy as np
from simple_logger import Logger

from thermography.utils.geometry import angle, angle_diff, segment_segment_intersection

__all__ = ["IntersectionDetector", "IntersectionDetectorParams"]


class IntersectionDetectorParams:
    """Parameters used by the :class:`.IntersectionDetector`."""

    def __init__(self):
        """Initializes the intersection detector parameters to their default value.

        :ivar angle_threshold: Only intersections between segments which deviate less than this parameter from the canonical 90° angle are accepted.
        """
        # All intersections between segments whose relative angle is larger than this threshold are ignored.
        self.angle_threshold = np.pi / 180 * 25


class IntersectionDetector:
    """Class responsible for detecting intersections between segments."""

    def __init__(self, input_segments: list, params: IntersectionDetectorParams = IntersectionDetectorParams()):
        """Initializes the intersection detector object.

        Intersections are computed between all segments `s_i` of cluster `i`, against all segments `s_j` of cluster `j`.

        :param input_segments: List of segment clusters. Each element `cluster_i` of this list is a numpy array of shape `[num_segments_i, 4]`
        :param params: Parameters to be used for intersection detection.
        """
        self.segments = input_segments
        self.params = params

        # Collection of intersections divided by clusters:
        # self.cluster_cluster_intersections[i,j] contains the intersections between cluster i and cluster j.
        self.cluster_cluster_intersections = {}
        self.raw_intersections = []

    def detect(self):
        """Detects the intersections between the segments passed to the constructor using the parameters passed to the
        constructor.

        .. note:: The intersections are only computed between segments belonging to different clusters, and never between segments of the same cluster.
        """
        Logger.debug("Detecting intersection")
        self.cluster_cluster_intersections = {}
        self.raw_intersections = []
        num_clusters = len(self.segments)
        for cluster_index_i in range(num_clusters):
            for cluster_index_j in range(cluster_index_i + 1, num_clusters):
                Logger.debug("Detecting intersections between cluster {} and cluster {}".format(cluster_index_i,
                                                                                                cluster_index_j))
                self.__detect_intersections_between_clusters(cluster_index_i, cluster_index_j)

    def __detect_intersections_between_clusters(self, cluster_index_i: int, cluster_index_j: int) -> None:
        """Detects the intersections between cluster `cluster_index_i` and cluster `cluster_index_j`.

        :param cluster_index_i: Index of first segment cluster to be intersected.
        :param cluster_index_j: Index of second segment cluster to be intersected.
        """
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
