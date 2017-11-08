import os

import cv2
import numpy as np
from simple_logger import Logger

from . import ModuleMap
from .classification import Inference
from .classification.models import ThermoNet3x3
from .detection import *
from .io import VideoLoader
from .settings import Camera, get_resources_dir
from .utils import aspect_ratio
from .utils.display import *


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
        Logger.debug("Starting thermo app")
        self.input_video_path = input_video_path
        self.camera_param_file = camera_param_file

        self.image_shape = np.array([96, 120, 1])
        self.num_classes = 3
        checkpoint_dir = os.path.join(get_resources_dir(), "weights")
        self.inference = Inference(checkpoint_dir=checkpoint_dir, model_class=ThermoNet3x3,
                                   image_shape=self.image_shape, num_classes=self.num_classes)

        # Camera object containing the corresponding parameters.
        self.camera = None

        # Object responsible for loading the video passed as parameter.
        self.video_loader = None

        # Global objects staying alive over the entire run of the movie.
        self.module_map = ModuleMap()
        self.motion_detector = MotionDetector(scaling=0.15)

        # Objects referring to the items computed during the last frame.
        self.last_input_frame = None
        self.last_preprocessed_image = None
        self.last_attention_image = None
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
        self.last_probabilities = {}

        # Runtime parameters for detection.
        self.should_undistort_image = True
        self.preprocessing_parameters = PreprocessingParams()
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
        Logger.debug("Creating segment image")
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
        Logger.debug("Creating rectangle image")
        base_image = self.last_scaled_frame_rgb.copy()
        mask = np.zeros_like(base_image)

        for module_id, module in self.module_map.global_module_map.items():
            if module.frame_id_history[-1] == self.last_frame_id:

                module_coords = module.last_rectangle - np.int32(module.cumulated_motion)
                mean_prob = module.mean_probability
                color = color_from_probabilities(mean_prob)

                cv2.polylines(base_image, np.int32([module_coords]), True, color, 1, cv2.LINE_AA)
                cv2.fillConvexPoly(mask, np.int32([module_coords]), color, cv2.LINE_4)
            else:
                continue

        cv2.addWeighted(base_image, 1.0, mask, 0.4, 0, base_image)
        return base_image

    def create_classes_image(self):
        Logger.debug("Creating classes image")
        base_image = self.last_scaled_frame_rgb.copy()

        for module_id, module in self.module_map.global_module_map.items():
            module_coords = module.last_rectangle - np.int32(module.cumulated_motion)
            module_center = module.last_center - np.int32(module.cumulated_motion)
            mean_prob = module.mean_probability
            color = color_from_probabilities(mean_prob)

            cv2.circle(base_image, (int(module_center[0]), int(module_center[1])), 6, color, cv2.FILLED, cv2.LINE_AA)
            cv2.polylines(base_image, np.int32([module_coords]), True, color, 1, cv2.LINE_AA)

        return base_image

    def create_module_map_image(self):
        Logger.debug("Creating module map image")
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
        Logger.debug("Creating module list")
        module_list = []
        module_width = 90
        module_height = 66
        padding = 15
        image_width = module_width + 2 * padding
        image_height = module_height + 2 * padding
        module_image_size = (image_width, image_height)

        for rectangle_id, rectangle in self.module_map.global_module_map.items():
            # Only iterate over the last detected rectangles.
            if rectangle.frame_id_history[-1] != self.last_frame_id:
                continue

            module_coordinates = rectangle.last_rectangle
            module_aspect_ratio = aspect_ratio(module_coordinates)
            is_horizontal = module_aspect_ratio >= 1.0
            if is_horizontal:
                projection_rectangle = np.float32([[0 + padding, 0 + padding],
                                                   [image_width - 1 - padding, 0 + padding],
                                                   [image_width - 1 - padding, image_height - 1 - padding],
                                                   [0 + padding, image_height - 1 - padding]])
            else:
                projection_rectangle = np.float32([[0 + padding, image_height - 1 - padding],
                                                   [0 + padding, 0 + padding],
                                                   [image_width - 1 - padding, 0 + padding],
                                                   [image_width - 1 - padding, image_height - 1 - padding]])

            transformation_matrix = cv2.getPerspectiveTransform(np.float32(module_coordinates),
                                                                projection_rectangle)
            extracted = cv2.warpPerspective(self.last_scaled_frame_rgb, transformation_matrix, module_image_size)

            module_list.append({"coordinates": rectangle.last_rectangle, "image": extracted, "id": rectangle.ID})

        return module_list

    def __load_params(self):
        """
        Load the parameters related to camera.
        """
        self.camera = Camera(camera_path=self.camera_param_file)

        Logger.info("Using camera parameters:\n{}".format(self.camera))

    def load_video(self, start_frame: int, end_frame: int):
        """
        Loads the video associated with the absolute path given to the constructor between the frames indicated as argument.

        :param start_frame: Starting frame (inclusive)
        :param end_frame: Termination frame (exclusive), is set to None, the entire video will be loaded.
        """
        self.video_loader = VideoLoader(video_path=self.input_video_path, start_frame=start_frame, end_frame=end_frame)

    def preprocess_frame(self):
        frame_preprocessor = FramePreprocessor(input_image=self.last_input_frame, params=self.preprocessing_parameters)
        frame_preprocessor.preprocess()

        self.last_scaled_frame_rgb = frame_preprocessor.scaled_image_rgb
        self.last_scaled_frame = frame_preprocessor.scaled_image
        self.last_preprocessed_image = frame_preprocessor.preprocessed_image
        self.last_attention_image = frame_preprocessor.attention_image

    def detect_edges(self):
        edge_detector = EdgeDetector(input_image=self.last_preprocessed_image, params=self.edge_detection_parameters)
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
        rectangle_detector = RectangleDetector(input_intersections=self.last_intersections,
                                               params=self.rectangle_detection_parameters)
        rectangle_detector.detect()
        self.last_rectangles = rectangle_detector.rectangles

    def classify_detected_modules(self):
        """
        Classifies the modules in the global module map which have been detected in the current frame. This function
        must be called after inserting the modules in the global module map!
        """
        assert (self.inference is not None)

        module_list = self.create_module_list()
        probabilities = self.inference.classify([m["image"] for m in module_list])
        for module, prob in zip(module_list, probabilities):
            self.last_probabilities[module["id"]] = prob

        self.module_map.update_class_belief(self.last_probabilities)

    def step(self, frame_id, frame):
        self.last_frame_id = frame_id
        self.last_input_frame = frame
        distorted_image = frame
        if self.should_undistort_image:
            undistorted_image = cv2.undistort(src=distorted_image, cameraMatrix=self.camera.camera_matrix,
                                              distCoeffs=self.camera.distortion_coeff)
        else:
            undistorted_image = distorted_image
        self.last_input_frame = undistorted_image

        self.preprocess_frame()

        self.detect_edges()
        self.detect_segments()
        if len(self.last_segments) < 3:
            Logger.warning("Found less than three segments!")
            return False

        self.cluster_segments()
        self.detect_intersections()
        self.detect_rectangles()

        # Motion estimate.
        self.last_mean_motion = self.motion_detector.motion_estimate(self.last_scaled_frame)

        # Add the detected rectangles to the global map.
        self.module_map.insert(self.last_rectangles, frame_id, self.last_mean_motion)

        if len(self.last_rectangles) == 0:
            Logger.warning("No rectangles detected!")
            return False

        return True

    def reset(self):
        self.last_input_frame = None
        self.last_preprocessed_image = None
        self.last_attention_image = None
        self.last_scaled_frame_rgb = None
        self.last_scaled_frame = None
        self.last_edges_frame = None
        self.last_raw_intersections = None
        self.last_intersections = None
        self.last_segments = None
        self.last_cluster_list = None
        self.last_rectangles = None
        self.last_mean_motion = None

        self.last_probabilities = {}

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
