import cv2
import numpy as np
import os
import progressbar
from .modes import Modality


class ImageLoader:
    def __init__(self, image_path, mode=None):
        """
        Initializes and loads the image associated to the file indicated by the path passed as argument.
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
            self.__mode = Modality.DEFAULT
        else:
            self.__mode = mode

    @property
    def image_raw(self):
        return self.__image_raw

    @image_raw.setter
    def image_raw(self, image_raw):
        self.__image_raw = image_raw


class VideoLoader:
    def __init__(self, video_path, start_frame=0, end_frame=None):
        """
        Loads the frames associated to the video indicated by the path passed as argument.
        :param video_path: Absolute path to the video to be loaded.
        :param start_frame: Start frame of the video to be considered (inclusive).
        :param end_frame: End frame of the video to be considered (non inclusive)
        """
        self.video_path = video_path

        self.start_frame = start_frame
        self.end_frame = end_frame

        self.frames = []
        self.__load_video(cv2.VideoCapture(self.video_path))

        cv2.VideoCapture()

    def show_video(self, fps=60):
        seconds_per_frame = 1.0 / fps
        for i, frame in enumerate(self.frames):
            cv2.imshow("Frame", frame)
            cv2.setWindowTitle("Frame", "Frame {}".format(self.start_frame + i))
            cv2.waitKey(int(1000 * seconds_per_frame))

    @property
    def video_path(self):
        return self.__video_path

    @video_path.setter
    def video_path(self, path):
        if not os.path.exists(path):
            raise FileExistsError("Video {} not found".format(self.video_path))
        self.__video_path = path

    def __load_video(self, video_raw: cv2.VideoCapture):
        if not video_raw.isOpened():
            print("Unable to read {} feed".format(self.video_path))

        self.frames = []

        num_video_frames = int(video_raw.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.end_frame is None:
            self.end_frame = num_video_frames

        num_frames = 0
        num_total_frames = self.end_frame - self.start_frame

        # Skip the first frames until the self_start frame.
        video_raw.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        print("Loading {} frames...".format(self.end_frame - self.start_frame))
        bar = progressbar.ProgressBar(maxval=num_total_frames,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i in range(self.end_frame - self.start_frame):
            if i + self.start_frame >= num_video_frames:
                RuntimeWarning(
                    "end_frame={} passed to VideoLoader is greater than video size {}."
                    "Closing video stream now.".format(self.end_frame, num_video_frames))
            ret = video_raw.grab()
            if not ret:
                raise ValueError("Could not load frame {}".format(i + self.start_frame))

            self.frames.append(video_raw.retrieve()[1])
            num_frames += 1
            bar.update(num_frames)

        bar.finish()
        video_raw.release()
