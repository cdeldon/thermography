import cv2
from enum import IntEnum, Enum


class Modality(IntEnum):
    """
    Modalities used to load an image into opencv.
    """

    RGB = cv2.IMREAD_COLOR
    GRAY_SCALE = cv2.IMREAD_GRAYSCALE

    # Set the default loading modality to RGB.
    DEFAULT = RGB


class LogLevel:
    """
    Log levels used for the simple_logger.
    """
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    WARN = WARNING
    ERROR = "ERROR"
    FATAL = "FATAL"
