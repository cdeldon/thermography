import cv2
import numpy as np

__all__ = ["rotate_image",
           "scale_image"]


def rotate_image(image: np.ndarray, a: float):
    """
    Rotates the input image by a radiants in counter-clock-wise direction.

    :param image: Image to be rotated.
    :param a: Rotation angle expressed in radiants.
    :return: Rotated version of the input image.
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, a / np.pi * 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def scale_image(input_image: np.ndarray, s: float):
    """
    Scales an input image by the value passed as parameter.

    :param input_image: Image to be scaled.
    :param s: Scalar value to be applied to the scaling procedure.
    :return: A copy of the input image scaled by the passed parameter.
    """
    return cv2.resize(src=input_image, dsize=(0, 0), fx=s, fy=s)
