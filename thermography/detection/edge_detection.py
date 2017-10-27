import cv2
import numpy as np
from simple_logger import Logger

__all__ = ["EdgeDetectorParams", "EdgeDetector"]


class EdgeDetectorParams:
    def __init__(self):
        self.hysteresis_min_thresh = 70
        self.hysteresis_max_thresh = 130

        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        self.dilation_steps = 4


class EdgeDetector:
    def __init__(self, input_image: np.ndarray, params: EdgeDetectorParams = EdgeDetectorParams()):
        self.input_image = input_image
        self.params = params

        self.edge_image = None

    def detect(self):
        """
        Detects the edges in the image passed to the constructor using the parameters passed to the constructor.
        """
        canny = cv2.Canny(image=self.input_image, threshold1=self.params.hysteresis_min_thresh,
                          threshold2=self.params.hysteresis_max_thresh, apertureSize=3)
        Logger.debug("Canny edges computed")

        dilated = cv2.dilate(canny, self.params.kernel,
                             iterations=self.params.dilation_steps)
        Logger.debug("Dilate canny edges with {} steps".format(self.params.dilation_steps))

        size = np.size(dilated)
        skel = np.zeros(dilated.shape, np.uint8)

        img = dilated
        done = False

        while not done:
            Logger.debug("Eroding canny edges")
            eroded = cv2.erode(img, self.params.kernel)
            temp = cv2.dilate(eroded, self.params.kernel)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        self.edge_image = skel
