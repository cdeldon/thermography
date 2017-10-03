import cv2
import numpy as np

__all__ = ["SegmentDetector", "SegmentDetectorParams"]


class SegmentDetectorParams:
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


class SegmentDetector:
    def __init__(self, input_image: np.ndarray, params=SegmentDetectorParams()):
        self.input_image = input_image
        self.params = params
        # Output "lines" is an array containing endpoints of detected line segments.
        self.segments = None

    def detect(self):
        # Run Hough on edge detected image.
        self.segments = cv2.HoughLinesP(image=self.input_image, rho=self.params.d_rho,
                                        theta=self.params.d_theta,
                                        threshold=self.params.min_num_votes,
                                        minLineLength=self.params.min_line_length,
                                        maxLineGap=self.params.max_line_gap)
        self.segments = np.squeeze(self.segments)
        if len(self.segments.shape) <= 1:
            self.segments = np.array([self.segments])
