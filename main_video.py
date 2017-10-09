import thermography as tg
from thermography.io import VideoLoader
from thermography.detection import *

import cv2
import numpy as np
import os

if __name__ == '__main__':

    # Load camera parameters.
    SETTINGS_DIR = tg.settings.get_settings_dir()
    camera_param_file = os.path.join(SETTINGS_DIR, "camera_parameters.json")
    camera = tg.settings.Camera(camera_path=camera_param_file)
    print("Using camera parameters:\n{}".format(camera))

    # Load module parameters.
    module_param_file = os.path.join(SETTINGS_DIR, "module_parameters.json")
    modules = tg.settings.Modules(module_path=module_param_file)
    print("Using module paramters:\n{}".format(modules))

    # Load data input parameters.
    THERMOGRAPHY_ROOT_DIR = tg.settings.get_thermography_root_dir()
    tg.settings.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/")
    IN_FILE_NAME = os.path.join(tg.settings.get_data_dir(), "Ispez Termografica Ghidoni 1.mov")

    # Input and preprocessing.
    video_loader = VideoLoader(video_path=IN_FILE_NAME, start_frame=1500, end_frame=1720)
    # video_loader.show_video(fps=25)

    # Global map of modules.
    module_map = tg.ModuleMap()

    motion_detector = MotionDetector(scaling=0.25)

    # cap = cv2.VideoCapture(1)
    # frame_id = -1
    # while True:
    #     frame_id += 1
    #     ret, frame = cap.read()

    for frame_id, frame in enumerate(video_loader.frames):
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
        edge_detector_params.dilation_steps = 4
        edge_detector_params.hysteresis_min_thresh = 60
        edge_detector_params.hysteresis_max_thresh = 180
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
        mean_angles, mean_centers = segment_clusterer.compute_cluster_mean()

        unfiltered_segments = segment_clusterer.cluster_list.copy()

        segment_clusterer.clean_clusters(mean_angles=mean_angles, max_angle_variation_mean=np.pi / 180 * 20,
                                         max_merging_angle=np.pi / 180 * 10, max_endpoint_distance=20.0)

        filtered_segments = segment_clusterer.cluster_list.copy()

        # Intersection detection
        intersection_detector_params = IntersectionDetectorParams()
        intersection_detector_params.angle_threshold = np.pi / 180 * 25
        intersection_detector = IntersectionDetector(input_segments=filtered_segments,
                                                     params=intersection_detector_params)
        intersection_detector.detect()

        # Detect the rectangles associated to the intersections.
        rectangle_detector_params = RectangleDetectorParams()
        rectangle_detector_params.aspect_ratio = modules.aspect_ratio
        rectangle_detector_params.aspect_ratio_relative_deviation = 0.5
        rectangle_detector = RectangleDetector(input_intersections=intersection_detector.cluster_cluster_intersections,
                                               params=rectangle_detector_params)
        rectangle_detector.detect()

        # Motion estimate.
        mean_motion = motion_detector.motion_estimate(gray)

        # Add the detected rectangles to the global map.
        # module_map.insert(rectangle_detector.rectangles, frame_id, mean_motion)

        # Displaying.
        base_image = cv2.cvtColor(src=gray, code=cv2.COLOR_GRAY2BGR)
        base_image = scaled_image
        tg.utils.draw_segments(segments=unfiltered_segments, base_image=base_image.copy(),
                               windows_name="Unfiltered segments", render_indices=False)
        tg.utils.draw_segments(segments=filtered_segments, base_image=base_image.copy(),
                               windows_name="Filtered segments")
        tg.utils.draw_intersections(intersections=intersection_detector.raw_intersections,
                                    base_image=base_image.copy(), windows_name="Intersections")
        tg.utils.draw_rectangles(rectangles=rectangle_detector.rectangles, base_image=base_image.copy(),
                                 windows_name="Detected rectangles")
        tg.utils.draw_motion(flow=motion_detector.flow, base_image=motion_detector.last_frame,
                             windows_name="Motion estimate")
        cv2.imshow("Canny edges", edge_detector.edge_image)

        global_map = base_image.copy()
        for rect_id, rectangle in module_map.global_rectangle_map.items():
            rect = rectangle.last_rectangle
            rect -= np.int32(rectangle.cumulated_motion)
            color = (0, 0, 255)
            cv2.polylines(global_map, np.int32([rect]), True, color, 2, cv2.LINE_4)
            center = np.mean(rect, axis=0)
            cv2.putText(global_map, str(rect_id), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 1)

        cv2.imshow("Global map", global_map)

        cv2.waitKey(1)

        # Rectangle extraction.
        # default_rect = np.float32([[629, 10], [10, 10], [10, 501], [629, 501]])
        # for rectangle in rectangle_detector.rectangles:
        #     M = cv2.getPerspectiveTransform(np.float32(rectangle), default_rect)
        #     extracted = cv2.warpPerspective(rectangle, M, (640, 512))

    print(module_map)
