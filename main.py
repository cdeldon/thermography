import termography as tg
from termography.io import ImageLoader
from termography.detection import LineDetector, LineDetectorParams

import cv2
import numpy as np
import os

if __name__ == '__main__':
    TERMOGRAPHY_ROOT_DIR = tg.get_termography_root_dir()
    tg.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/foto FLIR")
    IN_FILE_NAME = os.path.join(tg.get_data_dir(), "Hotspots.jpg")

    image_loader = ImageLoader(image_path=IN_FILE_NAME)

    gray = cv2.cvtColor(src=image_loader.image_raw, code=cv2.COLOR_BGR2GRAY)

    scale_factor = 1
    scaled = tg.scale(gray, scale_factor)

    canny = cv2.Canny(image=scaled, threshold1=40, threshold2=160, apertureSize=3)

    line_detector_params = LineDetectorParams()
    line_detector_params.min_line_length = 100
    line_detector_params.min_num_votes = 50
    line_detector_params.max_line_gap = 10
    line_detector = LineDetector(input_image=canny, line_detector_params=line_detector_params)
    line_detector.detect()

    edges = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    for line in line_detector.lines:
        cv2.line(img=edges, pt1=(line[0, 0], line[0, 1]), pt2=(line[0, 2], line[0, 3]),
                 color=(0, 180, 210), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow("Input", image_loader.image_raw)
    cv2.imshow("Canny", canny)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
