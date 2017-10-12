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

    def __init__(self, input_video_path, camera_param_file, module_param_file):
        """
        Initializes the ThermoApp object by defining default parameters.

        :param input_video_path: Absolute path to the input video.
        :param camera_param_file: Parameter file of the camera.
        :param module_param_file: Parameter file of the modules.
        """

        self.input_video_path = input_video_path
        self.camera_param_file = camera_param_file
        self.module_param_file = module_param_file

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
        self.last_rectangles = None
        self.last_mean_motion = None

        # Runtime parameters for detection.
        self.should_undistort_image = True
        self.image_rotating_angle = 0.0
        self.image_scaling = 1.0
        self.gaussian_blur = 3
        self.edge_detection_parameters = EdgeDetectorParams()
        self.segment_detection_parameters = SegmentDetectorParams()
        self.num_segment_clusters = 2
        self.swipe_clusters = False
        self.cluster_type = "gmm"
        self.intersection_detection_parameters = IntersectionDetectorParams()
        self.rectangle_detection_parameters = RectangleDetectorParams()

        # Load the camera and module parameters.
        self.__load_params()

    @property
    def frames(self):
        return self.video_loader.frames

    def __load_params(self):
        """
        Load the parameters related to camera and modules.
        """
        self.camera = Camera(camera_path=self.camera_param_file)
        self.modules = Modules(module_path=self.module_param_file)

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
        segment_clusterer = SegmentClusterer(input_segments=self.last_segments)
        segment_clusterer.cluster_segments(num_clusters=self.num_segment_clusters, n_init=8,
                                           cluster_type=self.cluster_type, swipe_clusters=self.swipe_clusters)

        mean_angles, mean_centers = segment_clusterer.compute_cluster_mean()
        segment_clusterer.clean_clusters(mean_angles=mean_angles, max_angle_variation_mean=np.pi / 180 * 20,
                                         max_merging_angle=np.pi / 180 * 10, max_endpoint_distance=10.0)

        self.last_segments = segment_clusterer.cluster_list

    def detect_intersections(self):
        intersection_detector = IntersectionDetector(input_segments=self.last_segments,
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

    def run(self):
        for frame_id, frame in enumerate(self.video_loader.frames):

            if self.step(frame_id, frame):
                # Displaying.
                base_image = self.last_scaled_frame_rgb

                draw_segments(segments=self.last_segments, base_image=base_image.copy(),
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
