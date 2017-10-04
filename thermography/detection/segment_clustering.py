import numpy as np
from matplotlib import pylab as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

import thermography as tg

__all__ = ["SegmentClusterer"]


class SegmentClusterer:
    def __init__(self, input_segments: np.ndarray):
        self.raw_segments = input_segments

        self.cluster_list = None
        self.cluster_features = None

    def cluster_segments(self, num_clusters: int = 15, n_init: int = 10, cluster_type: str = "gmm",
                         swipe_clusters: bool = True, use_angles: bool = True, use_centers: bool = False):
        """
        Clusters the input segments based on the parameters passed as argument. The features that can be used to cluster
        the segments are their mean coordinates, and their angle.
        :param num_clusters: Number of clusters to extract from the parameter space.
        :param n_init: Number of initializations to be performed when clustering.
        :param cluster_type: Clustering algorithm to be used, must be in ['gmm', 'knn'] which correspond to a full
        gaussian mixture model, and k-nearest-neighbors respectively.
        :param swipe_clusters: Boolean flag, if set to 'True' and 'cluster_type' is 'gmm', then the algorithm iterates
        the clustering procedure over a range of number of clusters from 1 to 'num_clusters' and retains the best
        result.
        :param use_angles: Boolean flag indicating whether to consider angles in the clustering process.
        :param use_centers: Boolean flag indicating whether to consider segment centroids in the clustering process.
        """
        if cluster_type not in ["gmm", "knn"]:
            raise ValueError("Invalid value for 'cluster_type': {} "
                             "'cluster_type' should be in ['gmm', 'knn']".format(cluster_type))

        centers = []
        angles = []
        for segment in self.raw_segments:
            pt1 = segment[0:2]
            pt2 = segment[2:4]
            center = (pt1 + pt2) * 0.5
            centers.append(center)

            # Segment angle lies in [0, pi], multiply by 2 such that complex number associated to similar angles are
            # close on the complex plane (e.g. 180° and 0°)
            angle = tg.utils.angle(pt1, pt2) * 2

            # Need to use complex representation as Euclidean distance used in clustering makes sense in complex plane,
            # and does not directly on angles.
            point = np.array([np.cos(angle), np.sin(angle)])
            angles.append(point)

        centers = np.array(centers)
        centers = normalize(centers, axis=0)
        angles = np.array(angles)

        if use_angles and use_centers:
            features = np.hstack((angles, centers))
        elif use_angles:
            features = angles
        elif use_centers:
            features = centers
        else:
            raise RuntimeError("Can not perform segment clustering without any feature. "
                               "Select 'use_angles=True' and/or 'use_centers=True'.")

        cluster_prediction = None

        if cluster_type is "knn":
            cluster_prediction = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=0).fit_predict(features)
        elif cluster_type is "gmm":
            best_gmm = None
            lowest_bic = np.infty
            bic = []
            n_components_range = range(1, num_clusters + 1)
            if not swipe_clusters:
                n_components_range = [num_clusters]
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM.
                gmm = GaussianMixture(n_components=n_components, covariance_type='full')
                gmm.fit(features)
                bic.append(gmm.bic(features))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

            cluster_prediction = best_gmm.predict(features)

        # Reorder the segments as clusters.
        cluster_segment_list = []
        cluster_feature_list = []
        num_labels = np.max(cluster_prediction) + 1
        for label in range(num_labels):
            cluster_segments = self.raw_segments[cluster_prediction == label]
            if len(cluster_segments) == 0:
                continue
            cluster_features = features[cluster_prediction == label]
            cluster_segment_list.append(cluster_segments)
            cluster_feature_list.append(cluster_features)

        self.cluster_list = cluster_segment_list
        self.cluster_features = cluster_feature_list

    def plot_segment_features(self):
        """
        Plots the first two dimensions of the features used for clustering.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for features in self.cluster_features:
            ax.scatter(features[:, 0], features[:, 1])

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

        plt.title('Segment clustering, {} components'.format(len(self.cluster_features)))
        plt.show()

    def compute_cluster_mean(self) -> tuple:
        """
        Computes the mean values (coordinates and angles) for each one of the identified clusters.
        :return: The mean angles, and mean coordinates of each cluster.
        """
        mean_centers = []
        mean_angles = []
        for cluster in self.cluster_list:
            centers = 0.5 * (cluster[:, 0:2] + cluster[:, 2:4])

            mean_center = np.mean(centers, axis=0)
            mean_centers.append(mean_center)

            mean_angles.append(tg.utils.mean_segment_angle(cluster))

        return np.array(mean_angles), np.array(mean_centers)

    def clean_clusters_angle(self, mean_angles: np.ndarray, max_angle_variation_mean: float):
        """
        Removes all segments whose angle deviates more than the passed parameter from the mean cluster angle.
        :param mean_angles: List of cluster means.
        :param max_angle_variation_mean: Maximal angle variation to allow between the cluster segments and the
        associated mean angle.
        """
        for cluster_index, (cluster, mean_angle) in enumerate(zip(self.cluster_list, mean_angles)):
            invalid_indices = []
            for segment_index, segment in enumerate(cluster):
                # Retrieve angle in [0, pi] of current segment.
                angle = tg.utils.angle(segment[0:2], segment[2:4])
                # Compute angle difference between current segment and mean angle of cluster.
                d_angle = tg.utils.angle_diff(angle, mean_angle)
                if d_angle > max_angle_variation_mean:
                    invalid_indices.append(segment_index)
            self.cluster_list[cluster_index] = np.delete(cluster, invalid_indices, axis=0)

    def merge_collinear_segments(self, max_merging_angle: float, max_endpoint_distance: float):
        """
        Merges all collinear segments belonging to the same cluster.
        :param max_merging_angle: Maximal angle to allow between segments to be merged.
        :param max_endpoint_distance: Maximal summed distance between segments endpoints and fitted line for merging
        segments.
        """
        for cluster_index, cluster in enumerate(self.cluster_list):
            merged = []
            merged_segments = []
            for i, segment_i in enumerate(cluster):
                if i in merged:
                    continue
                collinears = [i]
                for j in range(i + 1, len(cluster)):
                    segment_j = cluster[j]
                    if tg.utils.segments_collinear(segment_i, segment_j, max_angle=max_merging_angle,
                                                   max_endpoint_distance=max_endpoint_distance):
                        collinears.append(j)

                merged_segment = tg.utils.merge_segments(cluster[collinears])
                merged_segment = [int(m) for m in merged_segment]
                merged_segments.append(merged_segment)

                for index in collinears:
                    if index not in merged:
                        merged.append(index)

            self.cluster_list[cluster_index] = np.array(merged_segments)

    def clean_angle_consistency(self, max_neighbor_angle: float):
        """
        Removes the segments whose angle differs more than the passed threshold to the mean angle of its two neighbors.
        :param max_neighbor_angle: Maximal angle allowed between each segments and its two neighbors.
        If their angle differs more than this parameter, the central segment is discarded.
        """
        for cluster_index, cluster in enumerate(self.cluster_list):
            num_segments = len(cluster)
            invalid_indices = []
            for segment_index_i in range(0, num_segments):
                segment_i_min_1 = cluster[((segment_index_i - 1) + num_segments) % num_segments]
                segment_i = cluster[segment_index_i]
                segment_i_plus_1 = cluster[(segment_index_i + 1) % num_segments]
                angle_i_min_1 = tg.utils.angle(segment_i_min_1[0:2], segment_i_min_1[2:4])
                angle_i = tg.utils.angle(segment_i[0:2], segment_i[2:4])
                angle_i_plus_1 = tg.utils.angle(segment_i_plus_1[0:2], segment_i_plus_1[2:4])
                mean_angle_neighbors = np.mean([angle_i_min_1, angle_i_plus_1])
                if tg.utils.angle_diff(angle_i, mean_angle_neighbors) > max_neighbor_angle:
                    invalid_indices.append(segment_index_i)
            self.cluster_list[cluster_index] = np.delete(cluster, invalid_indices, axis=0)

    def clean_clusters(self, mean_angles: np.ndarray, max_angle_variation_mean: float = np.pi / 180 * 20,
                       max_merging_angle: float = 5.0 / 180 * np.pi, max_endpoint_distance: float = 50,
                       max_neighbor_angle=np.pi / 180 * 3):
        """
        Cleans the clusters by removing edges outliers (angle deviation from cluster mean is too high), and by merging
        almost collinear segments into a single segment.
        :param mean_angles: List of mean angles computed for each cluster.
        :param max_angle_variation_mean: Maximal allowed angle between each segment and corresponding cluster mean angle.
        :param max_merging_angle: Maximal allowed angle between two segments in order to merge them into a single one.
        :param max_endpoint_distance: Maximal summed distance between segments endpoints and fitted line for merging
        segments.
        :param max_neighbor_angle: Maximal angle allowed between each segments and its two neighbors.
        If their angle differs more than this parameter, the central segment is discarded.
        """

        # Reorder the segments inside the clusters.
        for cluster_index, (cluster, features) in enumerate(zip(self.cluster_list, self.cluster_features)):
            cluster_order = tg.utils.sort_segments(cluster)
            self.cluster_list[cluster_index] = cluster[cluster_order]
            self.cluster_features[cluster_index] = features[cluster_order]

        self.clean_clusters_angle(mean_angles=mean_angles, max_angle_variation_mean=max_angle_variation_mean)
        self.merge_collinear_segments(max_merging_angle=max_merging_angle, max_endpoint_distance=max_endpoint_distance)
        self.clean_angle_consistency(max_neighbor_angle=max_neighbor_angle)
