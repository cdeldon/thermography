import thermography as tg
from thermography.io import *
from thermography.detection import *

import cv2
import numpy as np
import os
import progressbar

if __name__ == '__main__':

    # Camera parameters.
    SETTINGS_DIR = tg.settings.get_settings_dir()
    camera_param_file = os.path.join(SETTINGS_DIR, "camera_parameters.json")
    camera = tg.settings.Camera(camera_path=camera_param_file)
    print("Using camera parameters:\n{}".format(camera))

    # Module parameters.
    module_param_file = os.path.join(SETTINGS_DIR, "module_parameters.json")
    modules = tg.settings.Modules(module_path=module_param_file)
    print("Using module paramters:\n{}".format(modules))

    # Data input parameters.
    THERMOGRAPHY_ROOT_DIR = tg.settings.get_thermography_root_dir()
    tg.settings.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/")
    IN_FILE_NAME = os.path.join(tg.settings.get_data_dir(), "Ispez Bellotti iniziale.mov")

    # Input and preprocessing.
    video_loader = VideoLoader(video_path=IN_FILE_NAME, start_frame=150, end_frame=2000)
    # video_loader.show_video(fps=25)

    bar = progressbar.ProgressBar(maxval=video_loader.num_frames,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for i, frame in enumerate(video_loader.frames):
        bar.update(i)
        frame = tg.utils.rotate_image(frame, np.pi * 0)
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
        segment_detector_params.min_line_length = 50
        segment_detector_params.min_num_votes = 50
        segment_detector_params.max_line_gap = 150
        segment_detector = SegmentDetector(input_image=edge_detector.edge_image, params=segment_detector_params)
        segment_detector.detect()

        if len(segment_detector.segments) < 3:
            continue

        # Segment clustering.
        segment_clusterer = SegmentClusterer(input_segments=segment_detector.segments)
        segment_clusterer.cluster_segments(num_clusters=2, n_init=8, cluster_type="gmm", swipe_clusters=False)
        # segment_clusterer.plot_segment_features()
        mean_angles, mean_centers = segment_clusterer.compute_cluster_mean()

        unfiltered_segments = segment_clusterer.cluster_list.copy()

        segment_clusterer.clean_clusters(mean_angles=mean_angles, max_angle_variation_mean=np.pi / 180 * 90,
                                         max_merging_angle=10.0 / 180 * np.pi, max_endpoint_distance=10.0)

        filtered_segments = segment_clusterer.cluster_list.copy()

        # Intersection detection
        intersection_detector = IntersectionDetector(input_segments=filtered_segments)
        intersection_detector.detect()

        # Detect the rectangles associated to the intersections.
        rectangle_detector_params = RectangleDetectorParams()
        rectangle_detector_params.aspect_ratio = modules.aspect_ratio
        rectangle_detector = RectangleDetector(input_intersections=intersection_detector.cluster_cluster_intersections,
                                               params=rectangle_detector_params)
        rectangle_detector.detect()

        # Displaying.
        edges = cv2.cvtColor(src=gray, code=cv2.COLOR_GRAY2BGR)
        edges_filtered = edges.copy()
        intersections = edges.copy()
        rectangles = edges.copy()

        # Fix colors for first two clusters, choose the next randomly.
        colors = [(29, 247, 240), (255, 180, 50)]
        for cluster_number in range(2, len(unfiltered_segments)):
            colors.append(tg.utils.random_color())

        for cluster, color in zip(unfiltered_segments, colors):
            for segment in cluster:
                cv2.line(img=edges, pt1=(segment[0], segment[1]), pt2=(segment[2], segment[3]),
                         color=color, thickness=1, lineType=cv2.LINE_AA)

        for cluster, color in zip(filtered_segments, colors):
            for segment_index, segment in enumerate(cluster):
                cv2.line(img=edges_filtered, pt1=(segment[0], segment[1]), pt2=(segment[2], segment[3]),
                         color=color, thickness=1, lineType=cv2.LINE_AA)
                cv2.putText(edges_filtered, str(segment_index), (segment[0], segment[1]), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)

        for intersection in intersection_detector.raw_intersections:
            cv2.circle(intersections, (int(intersection[0]), int(intersection[1])), 2, (0, 0, 255), 2, cv2.LINE_4)

        default_rect = np.float32([[629, 10], [10, 10], [10, 501], [629, 501]])
        for rectangle in rectangle_detector.rectangles:
            M = cv2.getPerspectiveTransform(np.float32(rectangle), default_rect)
            extracted = cv2.warpPerspective(rectangles, M, (640, 512))
            cv2.polylines(rectangles, np.int32([rectangle]), True, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("Skeleton", edge_detector.edge_image)
        cv2.imshow("Segments on input image", edges)
        cv2.imshow("Filtered segments on input image", edges_filtered)
        cv2.imshow("Intersections", intersections)
        cv2.imshow("Rectangles", rectangles)
        cv2.waitKey(1)
