import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import cmath

import thermography as tg

__all__ = ["SegmentClusterer"]


class SegmentClusterer:
    def __init__(self, input_segments: np.ndarray):
        self.raw_segments = input_segments

        self.cluster_list = None
        self.cluster_features = None

    def cluster_segments(self, num_clusters=15, n_init=10, cluster_type="gmm", swipe_clusters=True, use_angles=True,
                         use_centers=False):
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

        features = None
        if use_angles and use_centers:
            features = np.hstack((centers, angles))
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
            cluster_features = features[cluster_prediction == label]
            cluster_segment_list.append(cluster_segments)
            cluster_feature_list.append(cluster_features)

        self.cluster_list = cluster_segment_list
        self.cluster_features = cluster_feature_list

    def plot_segment_features(self):
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

    def compute_cluster_mean(self):
        mean_centers = []
        mean_angles = []
        for cluster in self.cluster_list:
            centers = 0.5 * (cluster[:, 0:2] + cluster[:, 2:4])

            mean_center = np.mean(centers, axis=0)
            mean_centers.append(mean_center)

            angles = []
            for segment in cluster:
                angle = tg.utils.angle(segment[0:2], segment[2:4]) * 2

                # Need to use complex representation as Euclidean distance used in clustering makes sense in
                # complex plane, and does not directly on angles.
                point = np.array([np.cos(angle), np.sin(angle)])
                angles.append(point)

            mean_point = np.mean(angles, axis=0)
            mean_angle = np.arctan2(mean_point[1], mean_point[0]) / 2
            mean_angles.append(mean_angle)

        return np.array(mean_angles), np.array(mean_centers)

    def clean_clusters_angle(self, mean_angles, max_angle_variation_mean):
        for cluster_index, (cluster, features, mean_angle) in enumerate(
                zip(self.cluster_list, self.cluster_features, mean_angles)):
            invalid_indices = []
            for segment_index, segment in enumerate(cluster):
                # Retrieve angle in [0, pi] of current segment.
                angle = tg.utils.angle(segment[0:2], segment[2:4])
                # Compute angle difference between current segment and mean angle of cluster.
                d_angle = tg.utils.angle_diff(angle, mean_angle)
                if d_angle > max_angle_variation_mean:
                    invalid_indices.append(segment_index)
            self.cluster_list[cluster_index] = np.delete(cluster, invalid_indices, axis=0)
            self.cluster_features[cluster_index] = np.delete(features, invalid_indices, axis=0)

    def merge_collinear_segments(self):
        for cluster_index, (cluster, features) in enumerate(zip(self.cluster_list, self.cluster_features)):
            found_collinear_segments = True
            while found_collinear_segments:
                found_collinear_segments = False
                for cluster_index, cluster in enumerate(self.cluster_list):
                    collinear_segments = []
                    for segment_index_i in range(len(cluster)):
                        segment_i = cluster[segment_index_i]
                        for segment_index_j in range(segment_index_i + 1, len(cluster)):
                            segment_j = cluster[segment_index_j]
                            slope, intercept = tg.line_estimate(segment_index_i, segment_index_j)

                    selected_indices = np.where(label == self.clusters)[0]

    def clean_clusters_too_close(self, min_intra_distance):
        for cluster_index, (cluster, features) in enumerate(zip(self.cluster_list, self.cluster_features)):
            invalid_indices = []
            for segment_index_i in range(len(cluster)):
                segment_i = cluster[segment_index_i, :]
                angle_i = tg.angle(segment_i[0:2], segment_i[2:4])
                for segment_index_j in range(segment_index_i + 1, len(cluster)):
                    segment_j = cluster[segment_index_j, :]
                    angle_j = tg.angle(segment_j[0:2], segment_j[2:4])

                    dist = tg.segment_min_distance(segment_i, segment_j)
                    if dist < min_intra_distance and np.abs(angle_i - angle_j) < np.pi / 180:
                        invalid_indices.append(segment_index_j)

            self.cluster_list[cluster_index] = np.delete(cluster, invalid_indices, axis=0)
            self.cluster_features[cluster_index] = np.delete(features, invalid_indices, axis=0)

    def clean_clusters(self, mean_angles, max_angle_variation_mean=np.pi / 180 * 20, min_intra_distance=0):

        self.clean_clusters_angle(mean_angles, max_angle_variation_mean)
        # self.merge_collinear_segments()
        # self.clean_clusters_too_close(min_intra_distance)
