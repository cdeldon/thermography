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
        self.min_num_votes = 60
        # Minimum number of pixels making up a line.
        self.min_line_length = 50
        # Maximum gap in pixels between connectible line segments.
        self.max_line_gap = 150
        # Number of pixels to extend each segment on each side.
        self.extension_pixels = 10


class SegmentDetector:
    def __init__(self, input_image: np.ndarray, params=SegmentDetectorParams()):
        self.input_image = input_image
        self.params = params
        # Output "lines" is an array containing endpoints of detected line segments.
        self.segments = None

    def __extend_segments(self):
        """
        Extends each segment by the parameters defined in self.params.extension_pixels.
        """
        dxs = self.segments[:, 2] - self.segments[:, 0]
        dys = self.segments[:, 3] - self.segments[:, 1]
        directions = np.vstack((dxs, dys)).T.astype(np.float32)
        norms = np.linalg.norm(directions, axis=1)
        directions /= norms[:, None]
        directions *= self.params.extension_pixels

        self.segments = np.int32(np.hstack((self.segments[:, 0:2] - directions, self.segments[:, 2:4] + directions)))

    def detect(self):
        """
        Detects the segments in the input image using the parameters passed as argument. Furthermore the detected
        segments are extended on each side by a few pixels as defined in the parameters.
        """
        self.segments = cv2.HoughLinesP(image=self.input_image, rho=self.params.d_rho,
                                        theta=self.params.d_theta,
                                        threshold=self.params.min_num_votes,
                                        minLineLength=self.params.min_line_length,
                                        maxLineGap=self.params.max_line_gap)

        # If no segments have been found, return an empty array.
        if self.segments is None:
            self.segments = np.empty(shape=(0, 4))
            return

        # Otherwise need to reshape the segment array, and to extend the segments on each side to allow better
        # intersection detection.
        self.segments = np.squeeze(self.segments)
        if len(self.segments.shape) <= 1:
            self.segments = np.array([self.segments])
        self.__extend_segments()
