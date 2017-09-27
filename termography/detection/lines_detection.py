import cv2
import matplotlib.pylab as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D


class LineDetectorParams:
    def __init__(self):
        # Distance resolution in pixels of the Hough grid.
        self.d_rho = 1.0
        # Angular resolution in radians of the Hough grid.
        self.d_theta = np.pi / 180
        # Minimum number of votes (intersections in Hough grid cell).
        self.min_num_votes = 15
        # Minimum number of pixels making up a line.
        self.min_line_length = 50
        # Maximum gap in pixels between connectible line segments.
        self.max_line_gap = 20


class LineDetector:
    def __init__(self, input_image: np.ndarray, line_detector_params=LineDetectorParams()):
        self.input_image = input_image
        self.line_detector_params = line_detector_params
        # Output "lines" is an array containing endpoints of detected line segments.
        self.lines = None
        self.clusters = None

    def detect(self):
        # Run Hough on edge detected image.
        self.lines = cv2.HoughLinesP(image=self.input_image, rho=self.line_detector_params.d_rho,
                                     theta=self.line_detector_params.d_theta,
                                     threshold=self.line_detector_params.min_num_votes,
                                     minLineLength=self.line_detector_params.min_line_length,
                                     maxLineGap=self.line_detector_params.max_line_gap)

    def cluster_lines(self, num_clusters=15, n_init=10):
        centers = []
        angles = []
        for line in self.lines:
            pt1 = line[0][0:2]
            pt2 = line[0][2:4]
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
        norm_features = normalize(features, axis=0)

        self.clusters = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=0).fit_predict(norm_features)

    def clean_clusters(self, max_line_distance):
        num_clusters = np.max(self.clusters)
        for label in range(num_clusters + 1):
            selected = self.lines[label == self.clusters]

    @property
    def input_image(self):
        return self.__input_image

    @input_image.setter
    def input_image(self, input_image):
        if len(input_image.shape) != 2:
            raise RuntimeError("Input image to {} must be a binary image".format(self.input_image.__name__))
        self.__input_image = input_image

    @property
    def lines(self):
        return self.__lines

    @lines.setter
    def lines(self, lines):
        self.__lines = lines
