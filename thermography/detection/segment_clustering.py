import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import thermography as tg


class SegmentClusterer:
    def __init__(self, input_segments: np.ndarray):
        self.segments = input_segments
        self.segment_features = None
        self.clusters = None

    def cluster_segments(self, num_clusters=15, n_init=10, cluster_type="gmm", swipe_clusters=True):
        if cluster_type not in ["gmm", "knn"]:
            raise ValueError("Invalid value for 'cluster_type': {} "
                             "'cluster_type' should be in ['gmm', 'knn']".format(cluster_type))

        centers = []
        angles = []
        for line in self.segments:
            pt1 = line[0:2]
            pt2 = line[2:4]
            center = (pt1 + pt2) * 0.5
            centers.append(center)

            if pt1[0] == pt2[0]:
                angles.append(np.pi * 0.5)
            else:
                angle = np.arctan((pt1[1] - pt2[1]) / (pt1[0] - pt2[0]))
                angles.append(angle)

        centers = np.array(centers)
        angles = np.array([angles])

        features = np.hstack((centers, angles.T))
        self.segment_features = normalize(features, axis=0)

        if cluster_type is "knn":
            self.clusters = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=0).fit_predict(
                self.segment_features)
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
                gmm.fit(self.segment_features)
                bic.append(gmm.bic(self.segment_features))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

            self.clusters = best_gmm.predict(self.segment_features)

    def plot_segment_features(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(np.max(self.clusters) + 1):
            ax.scatter(self.segment_features[self.clusters == i, 0], self.segment_features[self.clusters == i, 1],
                       self.segment_features[self.clusters == i, 2])

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_zlabel('Angle')

        plt.title('Segment clustering, {} components'.format(np.max(self.clusters) + 1))
        plt.show()

    def compute_cluster_mean(self):
        mean_centers = []
        mean_angles = []
        num_clusters = np.max(self.clusters)
        for label in range(num_clusters + 1):
            selection = self.segments[self.clusters == label]
            angles = []
            centers = []
            for segment in selection:
                angle = tg.angle(segment[0:2], segment[2:4])
                angles.append(angle)
                centers.append((segment[0:2] + segment[2:4]) * 0.5)
            mean_angle = np.median(angles)
            mean_angles.append(mean_angle)
            mean_center = np.mean(centers, axis=0)
            mean_centers.append(mean_center)
        return np.array(mean_angles), np.array(mean_centers)

    def clean_clusters_angle(self, mean_angles, max_angle_variation_mean):
        num_clusters = np.max(self.clusters)
        invalid_indices = []
        for label, mean_angle in zip(range(num_clusters + 1), mean_angles):
            selected_indices = np.where(label == self.clusters)[0]
            for index in selected_indices:
                segment = self.segments[index]
                angle = tg.angle(segment[0:2], segment[2:4])
                if np.abs(angle - mean_angle) > max_angle_variation_mean:
                    invalid_indices.append(index)
        self.clusters = np.delete(self.clusters, invalid_indices)
        self.segments = np.delete(self.segments, invalid_indices, axis=0)
        self.segment_features = np.delete(self.segment_features, invalid_indices, axis=0)

    def clean_clusters_too_close(self, min_intra_distance):
        num_clusters = np.max(self.clusters)
        invalid_indices = []
        for label in range(num_clusters + 1):
            selected_indices = np.where(label == self.clusters)[0]
            for _index_i in range(len(selected_indices)):
                index_i = selected_indices[_index_i]
                segment_i = self.segments[index_i]
                angle_i = tg.angle(segment_i[0:2], segment_i[2:4])
                for _index_j in range(_index_i+1, len(selected_indices)):
                    index_j = selected_indices[_index_j]
                    segment_j = self.segments[index_j]
                    angle_j = tg.angle(segment_j[0:2], segment_j[2:4])

                    dist = tg.segment_min_distance(segment_i, segment_j)
                    if dist < min_intra_distance and np.abs(angle_i - angle_j) < np.pi / 180 * 3:
                        invalid_indices.append(index_j)

        self.clusters = np.delete(self.clusters, invalid_indices)
        self.segments = np.delete(self.segments, invalid_indices, axis=0)
        self.segment_features = np.delete(self.segment_features, invalid_indices, axis=0)

    def clean_clusters(self, mean_angles, max_angle_variation_mean=np.pi / 180 * 20, max_intra_angle=0,
                       min_intra_distance=0):

        self.clean_clusters_angle(mean_angles, max_angle_variation_mean)
        self.clean_clusters_too_close(min_intra_distance)

