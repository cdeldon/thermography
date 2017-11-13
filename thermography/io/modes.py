from enum import IntEnum

import cv2


class Modality(IntEnum):
    """Modalities used to load an image into opencv."""

    RGB = cv2.IMREAD_COLOR
    GRAY_SCALE = cv2.IMREAD_GRAYSCALE

    # Set the default loading modality to RGB.
    DEFAULT = RGB


class LogLevel:
    """Log levels used for the :class:`simple_logger.Logger` object."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    WARN = WARNING
    ERROR = "ERROR"
    FATAL = "FATAL"
