import cv2
import numpy as np


def scale(input_image: np.ndarray, s: float):
    """
    Scales an input image by the value passed as parameter.
    :param input_image: Image to be scaled.
    :param s: Scalar value to be applied to the scaling procedure.
    :return: A copy of the input image scaled by the passed parameter.
    """
    return cv2.resize(src=input_image, dsize=(0, 0), fx=s, fy=s)
