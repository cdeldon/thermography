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
    tg.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/foto RGB")
    IN_FILE_NAME = os.path.join(tg.get_data_dir(), "DJI_0015.jpg")

    # Input preprocessing.
    image_loader = ImageLoader(image_path=IN_FILE_NAME)
    gray = cv2.cvtColor(src=image_loader.image_raw, code=cv2.COLOR_BGR2GRAY)

    scale_factor = 0.15
    gray = tg.scale_image(gray, scale_factor)
    gray = cv2.blur(gray, (3, 3))

    # Edge detection
    edge_detector_params = EdgeDetectorParams()
    edge_detector_params.dilation_steps = 1
    edge_detector_params.hysteresis_min_thresh = 40
    edge_detector_params.hysteresis_max_thresh = 120
    edge_detector = EdgeDetector(input_image=gray, params=edge_detector_params)
    edge_detector.detect()

    # Segment detection.
    segment_detector_params = SegmentDetectorParams()
    segment_detector_params.min_line_length = 150
    segment_detector_params.min_num_votes = 100
    segment_detector_params.max_line_gap = 35
    segment_detector = SegmentDetector(input_image=edge_detector.edge_image, params=segment_detector_params)
    segment_detector.detect()

    # Segment clustering.
    segment_clusterer = SegmentClusterer(input_segments=segment_detector.segments)
    segment_clusterer.cluster_segments(num_clusters=5, n_init=8, cluster_type="gmm")
    segment_clusterer.plot_segment_clusters()

    mean_angles, mean_centers = segment_clusterer.compute_cluster_mean()

    # Displaying.
    edges = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges_cleaned = edges.copy()

    colors = []
    for label in range(np.max(segment_clusterer.clusters) + 1):
        selected = segment_clusterer.segments[label == segment_clusterer.clusters]
        color = tg.random_color()
        colors.append(color)
        for line in selected:
            cv2.line(img=edges, pt1=(line[0], line[1]), pt2=(line[2], line[3]),
                     color=color, thickness=1, lineType=cv2.LINE_AA)

    segment_clusterer.clean_clusters(mean_angles=mean_angles, max_angle_variation_mean=np.pi / 180 * 20, min_intra_distance=5)
    for label, color in zip(range(np.max(segment_clusterer.clusters) + 1), colors):
        selected = segment_clusterer.segments[label == segment_clusterer.clusters]
        for line in selected:
            cv2.line(img=edges_cleaned, pt1=(line[0], line[1]), pt2=(line[2], line[3]),
                     color=color, thickness=1, lineType=cv2.LINE_AA)

    for angle, center, color in zip(mean_angles, mean_centers, colors):
        cv2.circle(img=edges, center=(int(center[0]), int(center[1])), radius=5, color=color,
                   thickness=-1, lineType=cv2.LINE_AA)
        slope = np.tan(angle)
        dir = np.array([1.0, 0.0])
        dir[1] = slope * dir[0]
        dir /= np.linalg.norm(dir)
        dir *= 25
        pt1 = center + dir
        pt2 = center - dir
        pt1 = pt1.astype(np.int)
        pt2 = pt2.astype(np.int)
        cv2.line(img=edges, pt1=(pt1[0], pt1[1]), pt2=(pt2[0], pt2[1]), color=color, thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img=edges_cleaned, pt1=(pt1[0], pt1[1]), pt2=(pt2[0], pt2[1]), color=color, thickness=2,
                 lineType=cv2.LINE_AA)

    cv2.imshow("Skeleton", edge_detector.edge_image)
    cv2.imshow("Segments on input image", edges)
    cv2.imshow("Cleaned segments on input image", edges_cleaned)
    cv2.waitKey(0)
