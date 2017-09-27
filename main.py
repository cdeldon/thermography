import termography as tg
from termography.io import ImageLoader
from termography.detection import EdgeDetector, EdgeDetectorParams, SegmentDetector, SegmentDetectorParams, \
    SegmentClusterer

import cv2
import numpy as np
import os

if __name__ == '__main__':

    # Data input parameters.
    TERMOGRAPHY_ROOT_DIR = tg.get_termography_root_dir()
    tg.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/foto FLIR")
    IN_FILE_NAME = os.path.join(tg.get_data_dir(), "Hotspots2.jpg")

    # Input preprocessing.
    image_loader = ImageLoader(image_path=IN_FILE_NAME)
    gray = cv2.cvtColor(src=image_loader.image_raw, code=cv2.COLOR_BGR2GRAY)

    scale_factor = 1
    scaled = tg.scale_image(gray, scale_factor)

    # Edge detection
    edge_detector_params = EdgeDetectorParams()
    edge_detector_params.dilation_steps = 2
    edge_detector_params.hysteresis_min_thresh = 40
    edge_detector_params.hysteresis_max_thresh = 160
    edge_detector = EdgeDetector(input_image=scaled, params=edge_detector_params)
    edge_detector.detect()

    # Segment detection.
    segment_detector_params = SegmentDetectorParams()
    segment_detector_params.min_line_length = 100
    segment_detector_params.min_num_votes = 40
    segment_detector_params.max_line_gap = 35
    segment_detector = SegmentDetector(input_image=edge_detector.edge_image, params=segment_detector_params)
    segment_detector.detect()

    # Segment clustering.
    segment_clusterer = SegmentClusterer(input_segments=segment_detector.segments)
    segment_clusterer.cluster_segments(num_clusters=25, n_init=5, cluster_type="gmm")
    segment_clusterer.plot_segment_clusters()

    # Displaying.
    edges = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for label in range(np.max(segment_clusterer.clusters) + 1):
        selected = segment_clusterer.segments[label == segment_clusterer.clusters]
        color = tg.random_color()
        for line in selected:
            cv2.line(img=edges, pt1=(line[0], line[1]), pt2=(line[2], line[3]),
                     color=color, thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow("Input", image_loader.image_raw)
    cv2.imshow("Skeleton", edge_detector.edge_image)
    cv2.imshow("Segments on input image", edges)
    cv2.waitKey(0)
