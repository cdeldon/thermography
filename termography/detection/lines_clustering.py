import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize


class LineClusterer:
    def __init__(self, input_lines: np.ndarray):
        self.lines = input_lines
        self.line_features = None
        self.clusters = None

    def cluster_lines(self, num_clusters=15, n_init=10, cluster_type="gmm"):
        if cluster_type not in ["gmm", "knn"]:
            raise ValueError("Invalid value for 'cluster_type': {} "
                             "'cluster_type' should be in ['gmm', 'knn']".format(cluster_type))

        centers = []
        angles = []
        for line in self.lines:
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
        self.line_features = normalize(features, axis=0)

        if cluster_type is "knn":
            self.clusters = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=0).fit_predict(
                self.line_features)
        elif cluster_type is "gmm":

            best_gmm = None
            lowest_bic = np.infty
            bic = []
            n_components_range = range(1, num_clusters + 1)
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM.
                gmm = GaussianMixture(n_components=n_components, covariance_type='full')
                gmm.fit(self.line_features)
                bic.append(gmm.bic(self.line_features))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

            self.clusters = best_gmm.predict(self.line_features)

    def plot_line_clusters(self):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(np.max(self.clusters) + 1):
            ax.scatter(self.line_features[self.clusters == i, 0], self.line_features[self.clusters == i, 1],
                       self.line_features[self.clusters == i, 2])

        plt.xticks(())
        plt.yticks(())
        plt.title('Line clustering, {} components'.format(np.max(self.clusters)))
        plt.show()

    def clean_clusters(self, max_line_distance):
        num_clusters = np.max(self.clusters)
        for label in range(num_clusters + 1):
            selected = self.lines[label == self.clusters]
