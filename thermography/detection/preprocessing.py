import cv2
import numpy as np

from thermography.utils import scale_image, rotate_image

__all__ = ["PreprocessingParams", "FramePreprocessor"]


class PreprocessingParams:
    def __init__(self):
        self.gaussian_blur = 2
        self.image_scaling = 1.0
        self.image_rotation = 0.0
        self.red_threshold = 200
        self.min_area = 60 * 60


class FramePreprocessor:
    def __init__(self, input_image: np.ndarray, params: PreprocessingParams = PreprocessingParams()):
        self.input_image = input_image
        self.params = params
        self.preprocessed_image = None
        self.scaled_image_rgb = None
        self.scaled_image = None
        self.attention_image = None

    @property
    def channels(self):
        if len(self.input_image.shape) < 3:
            return 1
        elif len(self.input_image.shape) == 3:
            return 3
        else:
            raise ValueError("Input image has {} channels.".format(len(self.input_image.shape)))

    @property
    def gray_scale(self):
        if self.channels == 1:
            return True
        elif self.channels == 3:
            return (self.input_image[:, :, 0] == self.input_image[:, :, 1]).all() and \
                   (self.input_image[:, :, 0] == self.input_image[:, :, 2]).all()
        else:
            raise ValueError("Input image has {} channels.".format(len(self.input_image.shape)))

    def preprocess(self) -> None:
        scaled_image = scale_image(self.input_image, self.params.image_scaling)
        rotated_frame = rotate_image(scaled_image, self.params.image_rotation)

        if self.params.gaussian_blur > 0:
            self.scaled_image = cv2.blur(self.scaled_image, (self.params.gaussian_blur, self.params.gaussian_blur))

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
