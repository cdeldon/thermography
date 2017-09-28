import thermography as tg
from thermography.io import VideoLoader

import cv2
import numpy as np
import os

if __name__ == '__main__':
    # Data input parameters.
    THERMOGRAPHY_ROOT_DIR = tg.get_thermography_root_dir()
    tg.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/")
    IN_FILE_NAME = os.path.join(tg.get_data_dir(), "Ispez Termografica Ghidoni 1.mov")

    # Input preprocessing.
    video_loader = VideoLoader(video_path=IN_FILE_NAME, start_frame=4000, end_frame=None)
    video_loader.show_video(fps=25)
