import os

import cv2
import progressbar
from simple_logger import Logger

from . import Modality

__all__ = ["ImageLoader", "VideoLoader"]


class ImageLoader:
    """Class responsible for loading a single image file into a numpy array."""

    def __init__(self, image_path: str, mode: Modality = Modality.DEFAULT):
        """Initializes and loads the image associated to the file indicated by the path passed as argument.

        :param image_path: Absolute path to the image file to be loaded.
        :param mode: Modality to be used when loading the image.
        """
        Logger.debug("Loading image at {}".format(image_path))
        self.image_path = image_path
        self.mode = mode
        self.image_raw = cv2.imread(self.image_path, self.mode)

    def show_raw(self, title: str = "", wait: int = 0) -> None:
        """Displays the raw image associated with the calling instance.

        :param title: Title to be added to the displayed image.
        :param wait: Time to wait until displayed windows is closed. If set to 0, then the image does not close.
        """
        cv2.imshow(title + " (raw)" if len(title) > 0 else "", self.image_raw)
        cv2.waitKey(wait)

    @property
    def image_path(self) -> str:
        """Returns the absolute path to the image loaded by this object."""
        return self.__image_path

    @image_path.setter
    def image_path(self, path: str):
        if not os.path.exists(path):
            raise FileExistsError("Image file {} not found".format(self.image_path))
        self.__image_path = path


class VideoLoader:
    """Class responsible for laoding a video into a sequence of numpy arrays representing the single video frames."""

    def __init__(self, video_path: str, start_frame: int = 0, end_frame: int = None):
        """Loads the frames associated to the video indicated by the path passed as argument.

        :param video_path: Absolute path to the video to be loaded.
        :param start_frame: Start frame of the video to be considered (inclusive).
        :param end_frame: End frame of the video to be considered (non inclusive). If set to None, the video will be loaded until the last frame.
        """
        Logger.debug("Loading video at {}".format(video_path))
        self.video_path = video_path

        self.start_frame = start_frame
        self.end_frame = end_frame
        Logger.debug("Start frame: {}, end frame: {}".format(self.start_frame, self.end_frame))

        self.frames = []
        self.__load_video(cv2.VideoCapture(self.video_path))

    @property
    def num_frames(self) -> int:
        """Returns the number of frames loaded by this object."""
        return self.end_frame - self.start_frame

    @property
    def video_path(self) -> str:
        """Returns the absolute path associated to the video loaded by this object."""
        return self.__video_path

    @video_path.setter
    def video_path(self, path: str):
        if not os.path.exists(path):
            Logger.fatal("Video path {} does not exist".format(path))
            raise FileNotFoundError("Video file {} not found".format(path))
        self.__video_path = path

    def __load_video(self, video_raw: cv2.VideoCapture):
        if not video_raw.isOpened():
            Logger.error("Unable to read {} feed".format(self.video_path))

        self.frames = []

        num_video_frames = int(video_raw.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.end_frame is None or self.end_frame > num_video_frames:
            Logger.warning("Setting end_frame to {}".format(num_video_frames))
            self.end_frame = num_video_frames

        num_frames = 0

        # Skip the first frames until the self_start frame.
        video_raw.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        Logger.info("Loading {} frames...".format(self.end_frame - self.start_frame))
        bar = progressbar.ProgressBar(maxval=self.num_frames,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in range(self.end_frame - self.start_frame):
            ret = video_raw.grab()
            if not ret:
                Logger.error("Could not load frame {}".format(i + self.start_frame))
                raise ValueError("Could not load frame {}".format(i + self.start_frame))

            self.frames.append(video_raw.retrieve()[1])
            num_frames += 1
            bar.update(num_frames)

        bar.finish()
        video_raw.release()
