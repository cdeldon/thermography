import cv2
import numpy as np

from thermography.utils import scale_image, rotate_image

__all__ = ["PreprocessingParams", "FramePreprocessor"]


class PreprocessingParams:
    """Parameters used by the :class:`.FramePreprocessor`."""

    def __init__(self):
        """Initializes the preprocessing parameters to their default value.

       :ivar gaussian_blur: Radius of the gaussian blur to apply to the input image.
       :ivar image_scaling: Scaling factor to apply to the input image.
       :ivar image_rotation: Angle expressed in radiants used to rotate the input image.
       :ivar red_threshold: Temperature threshold used to discard `cold` unimportant areas in the image.
       :ivar min_area: Minimal surface of retained `important` areas of the image. Warm regions whose surface is smaller than this threshold are discarded.
       """
        self.gaussian_blur = 1
        self.image_scaling = 1.0
        self.image_rotation = 0.0
        self.red_threshold = 200
        self.min_area = 60 * 60


class FramePreprocessor:
    """Class responsible for preprocessing an image frame."""

    def __init__(self, input_image: np.ndarray, params: PreprocessingParams = PreprocessingParams()):
        """Initializes the frame preprocessor with the input image and the preprocessor parameters.

        :param input_image: RGB or greyscale input image to be preprocessed.
        :param params: Preprocessing parameters.
        """
        self.input_image = input_image
        self.params = params
        self.preprocessed_image = None
        self.scaled_image_rgb = None
        self.scaled_image = None
        self.attention_image = None

    @property
    def channels(self) -> int:
        """Returns the number of channels of the :attr:`self.input_image` image."""
        if len(self.input_image.shape) < 3:
            return 1
        elif len(self.input_image.shape) == 3:
            return 3
        else:
            raise ValueError("Input image has {} channels.".format(len(self.input_image.shape)))

    @property
    def gray_scale(self) -> bool:
        """Returns a boolean indicating wheter :attr:`self.input_image` is a greyscale image (or an RGB image where all channels are identical)."""
        if self.channels == 1:
            return True
        elif self.channels == 3:
            return (self.input_image[:, :, 0] == self.input_image[:, :, 1]).all() and \
                   (self.input_image[:, :, 0] == self.input_image[:, :, 2]).all()
        else:
            raise ValueError("Input image has {} channels.".format(len(self.input_image.shape)))

    def preprocess(self) -> None:
        """Preprocesses the :attr:`self.input_image` following this steps:

            1. The image is scaled using the :attr:`self.params.image_scaling` parameter.
            2. The image is rotated using the :attr:`self.params.image_rotation` parameter.
            3. Attention detection.

                a. If the image is RGB, the :attr:`self.params.red_threshold` parameter is used to determine the attention areas of the image.
                b. Otherwise the entire image is kept as attention.

        """
        scaled_image = scale_image(self.input_image, self.params.image_scaling)
        rotated_frame = rotate_image(scaled_image, self.params.image_rotation)

        if self.params.gaussian_blur > 0:
            rotated_frame = cv2.blur(rotated_frame, (self.params.gaussian_blur, self.params.gaussian_blur))

        if self.channels == 1:
            self.scaled_image = rotated_frame
            self.scaled_image_rgb = cv2.cvtColor(self.scaled_image, cv2.COLOR_GRAY2BGR)
            self.preprocessed_image = self.scaled_image.astype(np.uint8)
            mask = np.ones_like(self.scaled_image).astype(np.uint8) * 255
        else:
            if self.gray_scale:
                self.scaled_image_rgb = rotated_frame
                self.scaled_image = rotated_frame[:, :, 0]
                self.preprocessed_image = self.scaled_image.astype(np.uint8)
                mask = np.ones_like(self.scaled_image).astype(np.uint8) * 255
            else:
                self.scaled_image_rgb = rotated_frame
                self.scaled_image = cv2.cvtColor(self.scaled_image_rgb, cv2.COLOR_BGR2GRAY)

                # Pixels with red channel larger or equal to params.red_threshold are colorcoded white in the binary image,
                # all other pixels are black.
                red_channel = self.scaled_image_rgb[:, :, 2]
                _, thresholded_image = cv2.threshold(red_channel, self.params.red_threshold, 255, 0, cv2.THRESH_BINARY)

                # Perform dilation and erosion on the thresholded image to remove holes and small islands.
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                closing = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
                opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

                # contours is a python list of all detected contours which are represented as numpy arrays of (x,y) coords.
                image, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                areas = [cv2.contourArea(contour) for contour in contours]
                discarded_contours = [area < self.params.min_area for area in areas]
                contours = [contours[i] for i in range(len(contours)) if not discarded_contours[i]]

                mask = np.zeros_like(self.scaled_image)
                cv2.drawContours(mask, contours, -1, (255), cv2.FILLED)

                mask = cv2.dilate(mask, kernel, iterations=5)
                mask = cv2.blur(mask, (25, 25))
                mask = mask.astype(np.float) / 255.
                self.preprocessed_image = (self.scaled_image * mask).astype(np.uint8)

                mask = (mask * 255).astype(np.uint8)

        attention_mask = cv2.applyColorMap(mask, cv2.COLORMAP_WINTER)
        self.attention_image = np.empty_like(self.scaled_image_rgb)
        cv2.addWeighted(cv2.cvtColor(self.scaled_image, cv2.COLOR_GRAY2BGR), 0.7, attention_mask, 0.3, 0,
                        self.attention_image)
