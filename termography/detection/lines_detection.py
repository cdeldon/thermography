import cv2
import matplotlib.pylab as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize


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

    def detect(self):
        # Run Hough on edge detected image.
        self.lines = cv2.HoughLinesP(image=self.input_image, rho=self.line_detector_params.d_rho,
                                     theta=self.line_detector_params.d_theta,
                                     threshold=self.line_detector_params.min_num_votes,
                                     minLineLength=self.line_detector_params.min_line_length,
                                     maxLineGap=self.line_detector_params.max_line_gap)

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
        if lines is not None:
            lines = np.squeeze(lines)
        self.__lines = lines
