import cv2
from enum import IntEnum


class Modality(IntEnum):
    """
    Modalities used to load an image into opencv.
    """

    RGB = cv2.IMREAD_COLOR
    GRAY_SCALE = cv2.IMREAD_GRAYSCALE

    # Set the default loading modality to RGB.
    DEFAULT = RGB
