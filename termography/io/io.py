import os
import cv2


class ImageLoader:
    def __init__(self, image_path, mode=None):
        """
        Initializes and loades the image associated to the file indicated by the path passed as argument.
        :param image_path: Absolute path to the image file to be loaded.
        :param mode: Modality to be used when laoding the image.
        """

        self.image_path = image_path
        self.mode = mode
        self.image_raw = cv2.imread(self.image_path, self.mode)

    def show_raw(self, title="", wait=0):
        """
        Displays the raw image associated with the calling instance.
        :param title: Title to be added to the displayed image.
        :param wait: Time to wait until displayed windows is closed. If set to 0, then the image does not close.
        """
        cv2.imshow(title + " (raw)" if len(title) > 0 else "", self.image_raw)
        cv2.waitKey(wait)

    @property
    def image_path(self):
        return self.__image_path

    @image_path.setter
    def image_path(self, path):
        if not os.path.exists(path):
            raise FileExistsError("Image {} not found".format(self.image_path))
        self.__image_path = path

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, mode):
        if mode is None:
            self.__mode = cv2.IMREAD_COLOR
        else:
            self.__mode = mode

    @property
    def image_raw(self):
        return self.__image_raw

    @image_raw.setter
    def image_raw(self, image_raw):
        self.__image_raw = image_raw
