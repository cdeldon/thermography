import thermography as tg
from thermography.io import *
from thermography.detection import *

import cv2
import numpy as np
import os

if __name__ == '__main__':

    # Camera parameters.
    SETTINGS_DIR = tg.settings.get_settings_dir()
    camera_param_file = os.path.join(SETTINGS_DIR, "camera_parameters.json")
    camera = tg.settings.Camera(camera_path=camera_param_file)

    print("Using camera parameters:\n{}".format(camera))

    # Data input parameters.
    THERMOGRAPHY_ROOT_DIR = tg.settings.get_thermography_root_dir()
    tg.settings.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/")
    IN_FILE_NAME = os.path.join(tg.settings.get_data_dir(), "Ispez Termografica Ghidoni 1.mov")

    # Input and preprocessing.
    video_loader = VideoLoader(video_path=IN_FILE_NAME, start_frame=1200, end_frame=1300)
    # video_loader.show_video(fps=25)

    for i, frame in enumerate(video_loader.frames):

        distorted_image = frame.copy()
        undistorted_image = cv2.undistort(src=distorted_image, cameraMatrix=camera.camera_matrix,
                                          distCoeffs=camera.distortion_coeff)

        scale_factor = 1.0
        scaled_image = tg.utils.scale_image(undistorted_image, scale_factor)

        gray = cv2.cvtColor(src=scaled_image, code=cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3, 3))

        # Edge detection.
        edge_detector_params = EdgeDetectorParams()
        edge_detector_params.dilation_steps = 2
        edge_detector_params.hysteresis_min_thresh = 30
        edge_detector_params.hysteresis_max_thresh = 100
        edge_detector = EdgeDetector(input_image=gray, params=edge_detector_params)
        edge_detector.detect()

        # Segment detection.
        segment_detector_params = SegmentDetectorParams()
        segment_detector_params.min_line_length = 150
        segment_detector_params.min_num_votes = 100
        segment_detector_params.max_line_gap = 250
        segment_detector = SegmentDetector(input_image=edge_detector.edge_image, params=segment_detector_params)
        segment_detector.detect()

        # Segment clustering.
        segment_clusterer = SegmentClusterer(input_segments=segment_detector.segments)
        segment_clusterer.cluster_segments(num_clusters=2, n_init=8, cluster_type="gmm", swipe_clusters=False)
        # segment_clusterer.plot_segment_features()
        mean_angles, mean_centers = segment_clusterer.compute_cluster_mean()

        unfiltered_segments = segment_clusterer.cluster_list.copy()

        segment_clusterer.clean_clusters(mean_angles=mean_angles, max_angle_variation_mean=np.pi / 180 * 20,
                                         max_merging_angle=10.0 / 180 * np.pi, max_endpoint_distance=10.0)

        filtered_segments = segment_clusterer.cluster_list.copy()

        # Intersection detection
        intersection_detector = IntersectionDetector(input_segments=filtered_segments)
        intersection_detector.detect()

        # Displaying.
        edges = cv2.cvtColor(src=gray, code=cv2.COLOR_GRAY2BGR)
        edges_filtered = edges.copy()

        # Fix colors for first two clusters, choose the next randomly.
        colors = [(29, 247, 240), (255, 180, 50)]
        for cluster_number in range(2, len(unfiltered_segments)):
            colors.append(tg.utils.random_color())

        for cluster, color in zip(unfiltered_segments, colors):
            for segment in cluster:
                cv2.line(img=edges, pt1=(segment[0], segment[1]), pt2=(segment[2], segment[3]),
                         color=color, thickness=1, lineType=cv2.LINE_AA)

        for cluster, color in zip(filtered_segments, colors):
            for segment in cluster:
                cv2.line(img=edges_filtered, pt1=(segment[0], segment[1]), pt2=(segment[2], segment[3]),
                         color=color, thickness=1, lineType=cv2.LINE_AA)

        for intersection in intersection_detector.raw_intersections:
            cv2.circle(edges_filtered, (int(intersection[0]), int(intersection[1])), radius=1, color=(0, 0, 255),
                       thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow("Skeleton", edge_detector.edge_image)
        cv2.imshow("Segments on input image", edges)
        cv2.imshow("Filtered segments on input image", edges_filtered)
        cv2.waitKey(1)
