import termography as tg
from termography.io import ImageLoader
from termography.detection import LineDetector, LineDetectorParams, LineClusterer

import cv2
import numpy as np
import os

if __name__ == '__main__':

    # Data input parameters.
    TERMOGRAPHY_ROOT_DIR = tg.get_termography_root_dir()
    tg.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/foto FLIR")
    IN_FILE_NAME = os.path.join(tg.get_data_dir(), "Hotspots.jpg")

    # Input preprocessing.
    image_loader = ImageLoader(image_path=IN_FILE_NAME)
    gray = cv2.cvtColor(src=image_loader.image_raw, code=cv2.COLOR_BGR2GRAY)

    scale_factor = 1
    scaled = tg.scale_image(gray, scale_factor)

    # Line detection.
    canny = cv2.Canny(image=scaled, threshold1=40, threshold2=160, apertureSize=3)

    line_detector_params = LineDetectorParams()
    line_detector_params.min_line_length = 100
    line_detector_params.min_num_votes = 50
    line_detector_params.max_line_gap = 10
    line_detector = LineDetector(input_image=canny, line_detector_params=line_detector_params)
    line_detector.detect()

    # Line clustering.
    line_clusterer = LineClusterer(input_lines=line_detector.lines)
    line_clusterer.cluster_lines(num_clusters=20, n_init=5)
    line_clusterer.plot_line_clusters()

    # Displaying.
    edges = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    for label in range(np.max(line_clusterer.clusters) + 1):
        selected = line_clusterer.lines[label == line_clusterer.clusters]
        color = tg.random_color()
        for line in selected:
            cv2.line(img=edges, pt1=(line[0], line[1]), pt2=(line[2], line[3]),
                     color=color, thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow("Input", image_loader.image_raw)
    cv2.imshow("Canny", canny)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
