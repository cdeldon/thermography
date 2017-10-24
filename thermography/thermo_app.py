from . import ModuleMap
from .io import VideoLoader
from .detection import *
from .settings import Camera, Modules
from .utils import rotate_image, scale_image
from .utils.display import *

import cv2
import numpy as np


class ThermoApp:
    """
    Application implementing the routines for module detections and analysis under the form of an object.
    """

    def __init__(self, input_video_path, camera_param_file):
        """
        Initializes the ThermoApp object by defining default parameters.

        :param input_video_path: Absolute path to the input video.
        :param camera_param_file: Parameter file of the camera.
        """

        self.input_video_path = input_video_path
        self.camera_param_file = camera_param_file

        # Camera and Modules object containing the corresponding parameters.
        self.camera = None
        self.modules = None

        # Object responsible for loading the video passed as parameter.
        self.video_loader = None

        # Global objects staying alive over the entire run of the movie.
        self.module_map = ModuleMap()
        self.motion_detector = MotionDetector(scaling=0.15)

        # Objects referring to the items computed during the last frame.
        self.last_input_frame = None
        self.last_scaled_frame_rgb = None
        self.last_scaled_frame = None
        self.last_edges_frame = None
        self.last_raw_intersections = None
        self.last_intersections = None
        self.last_segments = None
        self.last_cluster_list = None
        self.last_rectangles = None
        self.last_mean_motion = None
        self.last_frame_id = 0

        # Runtime parameters for detection.
        self.should_undistort_image = True
        self.image_rotating_angle = 0.0
        self.image_scaling = 1.0
        self.gaussian_blur = 3
        self.edge_detection_parameters = EdgeDetectorParams()
        self.segment_detection_parameters = SegmentDetectorParams()
        self.segment_clustering_parameters = SegmentClustererParams()
        self.cluster_cleaning_parameters = ClusterCleaningParams()
        self.intersection_detection_parameters = IntersectionDetectorParams()
        self.rectangle_detection_parameters = RectangleDetectorParams()

        # Load the camera and module parameters.
        self.__load_params()

    @property
    def frames(self):
        return self.video_loader.frames

    def create_segment_image(self):
        base_image = self.last_scaled_frame_rgb.copy()
        if self.last_cluster_list is None:
            return base_image

        # Fix colors for first two clusters, choose the next randomly.
        colors = [(29, 247, 240), (255, 180, 50)]
        for cluster_number in range(2, len(self.last_cluster_list)):
            colors.append(random_color())

        for cluster, color in zip(self.last_cluster_list, colors):
            for segment_index, segment in enumerate(cluster):
                cv2.line(img=base_image, pt1=(segment[0], segment[1]), pt2=(segment[2], segment[3]),
                         color=color, thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(base_image, str(segment_index), (segment[0], segment[1]), cv2.FONT_HERSHEY_PLAIN, 1.7,
                            (255, 255, 255), 2)
        return base_image

    def create_rectangle_image(self):
        base_image = self.last_scaled_frame_rgb.copy()
        if self.last_rectangles is not None and len(self.last_rectangles) > 0:
            mean_color = np.mean(base_image, axis=(0, 1))
            mask = np.zeros_like(base_image)
            if mean_color[0] == mean_color[1] == mean_color[2]:
                mean_color = np.array([255, 255, 0])
            opposite_color = np.array([255, 255, 255]) - mean_color
            opposite_color = (int(opposite_color[0]), int(opposite_color[1]), int(opposite_color[2]))
            for rectangle in self.last_rectangles:
                cv2.polylines(base_image, np.int32([rectangle]), True, opposite_color, 1, cv2.LINE_AA)
                cv2.fillConvexPoly(mask, np.int32([rectangle]), (255, 0, 0), cv2.LINE_4)

            cv2.addWeighted(base_image, 1, mask, 0.3, 0, base_image)
        return base_image

    def create_module_map_image(self):
        base_image = self.last_scaled_frame_rgb.copy()
        for rect_id, rectangle in self.module_map.global_module_map.items():
            rect_shift = rectangle.last_rectangle - np.int32(rectangle.cumulated_motion)
            if rectangle.frame_id_history[-1] == self.last_frame_id:
                color = (0, 0, 255)
                thickness = 2
            else:
                color = (255, 0, 0)
                thickness = 1
            cv2.polylines(base_image, np.int32([rect_shift]), True, color, thickness, cv2.LINE_AA)
            center = np.mean(rect_shift, axis=0)
            if thickness > 1:
                cv2.putText(base_image, str(rect_id), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_PLAIN,
                            1, (255, 255, 255), 1)

        return base_image

    def create_module_list(self):
        module_list = []
        module_shape = (90, 64)
        default_rect = np.float32(
            [[module_shape[0] - 1, 0], [0, 0], [0, module_shape[1] - 1], [module_shape[0] - 1, module_shape[1] - 1]])
        for rectangle_id, rectangle in self.module_map.global_module_map.items():
            if rectangle.frame_id_history[-1] != self.last_frame_id:
                continue

            M = cv2.getPerspectiveTransform(np.float32(rectangle.last_rectangle), default_rect)
            extracted = cv2.warpPerspective(self.last_scaled_frame_rgb, M, module_shape)
            module_list.append({"coordinates": rectangle.last_rectangle, "image": extracted, "id": rectangle.ID})
        return module_list

    def __load_params(self):
        """
        Load the parameters related to camera and modules.
        """
        self.camera = Camera(camera_path=self.camera_param_file)
        self.modules = Modules()

        print("Using camera parameters:\n{}".format(self.camera))
        print()
        print("Using module parameters:\n{}".format(self.modules))
        print()

    def load_video(self, start_frame: int, end_frame: int):
        """
        Loads the video associated with the absolute path given to the constructor between the frames indicated as argument.

        :param start_frame: Starting frame (inclusive)
        :param end_frame: Termination frame (exclusive), is set to None, the entire video will be loaded.
        """
        self.video_loader = VideoLoader(video_path=self.input_video_path, start_frame=start_frame, end_frame=end_frame)

    def detect_edges(self):
        edge_detector = EdgeDetector(input_image=self.last_scaled_frame, params=self.edge_detection_parameters)
        edge_detector.detect()

        self.last_edges_frame = edge_detector.edge_image

    def detect_segments(self):
        segment_detector = SegmentDetector(input_image=self.last_edges_frame, params=self.segment_detection_parameters)
        segment_detector.detect()

        self.last_segments = segment_detector.segments

    def cluster_segments(self):
        segment_clusterer = SegmentClusterer(input_segments=self.last_segments,
                                             params=self.segment_clustering_parameters)
        segment_clusterer.cluster_segments()

        mean_angles, mean_centers = segment_clusterer.compute_cluster_mean()
        segment_clusterer.clean_clusters(mean_angles=mean_angles, params=self.cluster_cleaning_parameters)

        self.last_cluster_list = segment_clusterer.cluster_list

    def detect_intersections(self):
        intersection_detector = IntersectionDetector(input_segments=self.last_cluster_list,
                                                     params=self.intersection_detection_parameters)
        intersection_detector.detect()
        self.last_raw_intersections = intersection_detector.raw_intersections
        self.last_intersections = intersection_detector.cluster_cluster_intersections

    def detect_rectangles(self):
        self.rectangle_detection_parameters.aspect_ratio = self.modules.aspect_ratio
        rectangle_detector = RectangleDetector(input_intersections=self.last_intersections,
                                               params=self.rectangle_detection_parameters)
        rectangle_detector.detect()
        self.last_rectangles = rectangle_detector.rectangles

    def step(self, frame_id, frame):
        self.last_frame_id = frame_id
        self.last_input_frame = frame
        distorted_image = frame
        if self.should_undistort_image:
            undistorted_image = cv2.undistort(src=distorted_image, cameraMatrix=self.camera.camera_matrix,
                                              distCoeffs=self.camera.distortion_coeff)
        else:
            undistorted_image = distorted_image

        scaled_image = scale_image(undistorted_image, self.image_scaling)

        rotated_frame = rotate_image(scaled_image, self.image_rotating_angle)
        self.last_scaled_frame_rgb = rotated_frame

        gray = cv2.cvtColor(src=rotated_frame, code=cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (self.gaussian_blur, self.gaussian_blur))

        self.last_scaled_frame = gray

        self.detect_edges()
        self.detect_segments()
        if len(self.last_segments) < 3:
            return False

        self.cluster_segments()
        self.detect_intersections()
        self.detect_rectangles()

        # Motion estimate.
        self.last_mean_motion = self.motion_detector.motion_estimate(self.last_scaled_frame)

        # Add the detected rectangles to the global map.
        self.module_map.insert(self.last_rectangles, frame_id, self.last_mean_motion)

        return True

    def reset(self):
        self.last_input_frame = None
        self.last_scaled_frame_rgb = None
        self.last_scaled_frame = None
        self.last_edges_frame = None
        self.last_raw_intersections = None
        self.last_intersections = None
        self.last_segments = None
        self.last_cluster_list = None
        self.last_rectangles = None
        self.last_mean_motion = None

    def run(self):
        for frame_id, frame in enumerate(self.video_loader.frames):

            if self.step(frame_id, frame):
                # Displaying.
                base_image = self.last_scaled_frame_rgb

                draw_segments(segments=self.last_cluster_list, base_image=base_image.copy(),
                              windows_name="Filtered segments")
                draw_intersections(intersections=self.last_raw_intersections,
                                   base_image=base_image.copy(), windows_name="Intersections")
                draw_rectangles(rectangles=self.last_rectangles, base_image=base_image.copy(),
                                windows_name="Detected rectangles")
                draw_motion(flow=self.motion_detector.flow, base_image=self.motion_detector.last_frame,
                            windows_name="Motion estimate")
                cv2.imshow("Canny edges", self.last_edges_frame)

                global_map = base_image.copy()
                for rect_id, rectangle in self.module_map.global_module_map.items():
                    rect_shift = rectangle.last_rectangle - np.int32(rectangle.cumulated_motion)
                    if rectangle.frame_id_history[-1] == frame_id:
                        color = (0, 0, 255)
                        thickness = 2
                    else:
                        color = (255, 0, 0)
                        thickness = 1
                    cv2.polylines(global_map, np.int32([rect_shift]), True, color, thickness, cv2.LINE_AA)
                    center = np.mean(rect_shift, axis=0)
                    if thickness > 1:
                        cv2.putText(global_map, str(rect_id), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_PLAIN,
                                    1,
                                    (255, 255, 255), 1)

                cv2.imshow("Global map", global_map)

                cv2.waitKey(1)
            self.reset()
