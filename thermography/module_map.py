import numpy as np
from simple_logger import Logger

from thermography.utils import ID, rectangle_contains, area_between_rectangles, area


class ModuleMap:
    """Class responsible for storing a spatial representation of the detected modules.
    Each module is keyed by an unique ID Multiple detections of the same module are considered and the existing module is updated in the internal representation of the map.

    :Example:

    .. code-block:: python

        module_map = ModuleMap()
        module_map.insert([rect1, rect2, ...], frame_id=0, motion_estimate = np.array([0, 0])
        # module_map.global_module_map = {0: <..., rect1, frame_id_history=[0]>;
        #                                 1: <..., rect2, frame_id_history=[0]>,
        #                                 2: <..., rect3, frame_id_history=[0]>}
        motion = np.array([4, 2])
        rect1 += motion
        module_map.insert([rect1, rect4], frame_id=1, motion_estimate=motion)
        # module_map.global_module_map = {0: <..., rect1, frame_id_history=[0, 1]>;
        #                                 1: <..., rect2, frame_id_history=[0]>,
        #                                 2: <..., rect3, frame_id_history=[0]>,
        #                                 3: <..., rect4, frame_id_history=[1]>}

    """

    class __ModuleInMap:
        """Class representing a single module instance.

        The module's property are given by a unique ID, the image-coordinates of the last detection,
        the image-coordinates of the module center during the last detection, the module's surface,
        a list of frames in which the module has been detected, and a list of associated image-coordinates.

        An estimate of the motion occurred from the last time the module has been detected and the current frame is
        An estimate of the motion occurred from the last time the module has been detected and the current frame is
        updated each time new modules are added to the module map.

        An estimate of class probabilities associated to the module is computed by taking the mean over all probability
        samples associated to the module.
        """

        def __init__(self, ID: int, rectangle: np.ndarray, frame_id: int):
            """Initializes the module with the ID, image-coordinates and frame id passed as argument.

            :param ID: Unique integer identifier. This module is keyed by this parameter in the ModuleMap.
            :param rectangle: Numpy array of pixel-coordinates associated to the first module detection of the form np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
            :param frame_id: Integer associated to the video-frame the module has been detected from.
            """
            Logger.debug("Creating a new module inside the map with ID {}".format(ID))
            self.ID = ID
            self.last_rectangle = None
            self.last_center = None
            self.last_area = None

            # Initialize the frame history and rectangle history to be empty. They will be filled in the self.add(...) method.
            self.frame_id_history = []
            self.rectangle_history = {}

            # Since the module has been detected now, the motion estimate is set to zero.
            self.cumulated_motion = np.array([0, 0], dtype=np.float32)
            self.__all_probabilities = []

            self.add(rectangle, frame_id)

        def __repr__(self):
            s = ""
            s += "Frame: {},  center: {}\n\t".format(self.frame_id_history[-1], self.last_center) + str(
                self.last_rectangle).replace('\n', '\n\t')
            s += "\n\tHistory: "
            if len(self.frame_id_history) == 1:
                s += "{}"
            else:
                for frame_id in self.frame_id_history[-2::-1]:
                    s += "\n\t\t({}): ".format(frame_id)
                    s += str(self.rectangle_history[frame_id]).replace('\n', '\n\t\t\t')

            return s

        def add(self, rectangle: np.ndarray, frame_id: int) -> None:
            """Adds a the observation of the module to the current state.

            :param rectangle: New image-coordinates of the module being observed.
            :param frame_id: Frame id of the frame in which the module has been detected.
            """
            self.last_rectangle = rectangle
            self.last_center = np.mean(self.last_rectangle, axis=0)
            self.last_area = area(self.last_rectangle)

            self.frame_id_history.append(frame_id)
            self.rectangle_history[frame_id] = rectangle

            # Since the module has been detected now, the motion estimate is reset to zero.
            self.cumulated_motion = np.array([0, 0], dtype=np.float32)

        def add_motion(self, frame_id: int, motion_estimate: np.ndarray) -> None:
            """Add a motion estimate to the module.

            Since a module might not be detected during one frame, the module map shifts its coordinates according to a
            motion estimate of the frame. This motion estimate is propagated to the modules which are not detected i
            the current frame in order to move their coordinates with the image.

            :param frame_id: Current video-frame.
            :param motion_estimate: Motion estimate between frame_id-1 and frame_id expressed in pixel coordinates.
            """

            # Only update the motion estimate if the current module has not been detected in this frame.
            if frame_id != self.frame_id_history[-1]:
                self.cumulated_motion += motion_estimate

        def update_probability(self, prob: np.ndarray) -> None:
            """Updates the current probability distribution over the class labels of this module.

            :param prob: A 1D numpy array of size 'num_classes' representing the classification probability.
            """
            self.__all_probabilities.append(prob)

        @property
        def mean_probability(self) -> np.ndarray:
            """Computes the mean class probability for the current module."""
            if len(self.__all_probabilities) == 0:
                raise RuntimeError("No probabilities assigned to current module {}".format(self.ID))
            return np.mean(self.__all_probabilities, axis=0)

    def __init__(self):
        """Initializes the module map to be empty."""
        Logger.debug("Creating the module map")
        # A dictionary of modules and their centers keyed by their ID.
        self.global_module_map = {}
        self.module_database = []

    def __repr__(self):
        s = ""
        for module_id, module_in_map in self.global_module_map.items():
            s += str(module_in_map) + "\n\n"
        return s

    def insert(self, rectangle_list: list, frame_id: int, motion_estimate: np.ndarray = None):
        """Inserts all rectangles contained in the list  passed as first parameter into the global map representation.

        :param rectangle_list: List of detected rectangles to be inserted into the global module map.
        :param frame_id: Frame id of the current image frame associated to the detected rectangles.
        :param motion_estimate: Numpy array representing the motion estimate between the last frame (ID-1) and the frame containing the rectangles.
        """

        Logger.debug("Inserting a new rectangle list into the module map at frame {}".format(frame_id))
        # When no information about the motion estimate is given, assume no motion.
        if motion_estimate is None:
            motion_estimate = np.array([0.0, 0.0])

        # In case there are no rectangles in the global map (first step in the simulation) store all the ones passed to
        # the function as if they were new modules.
        if len(self.global_module_map) == 0:
            for rectangle in rectangle_list:
                # Give a new ID to each rectangle.
                next_ID = ID.next_id()
                self.global_module_map[next_ID] = self.__ModuleInMap(next_ID, rectangle, frame_id)
        else:
            # Correspondences between the rectangles passed to the function and the ones already stored in the global map.
            # If no correspondence is found (e.g. a rectangle is new), then set the correspondence to None.
            correspondences = []
            for rectangle in rectangle_list:
                rectangle_center = np.mean(rectangle, axis=0)
                # Shift the rectangle center using the motion estimate in order to align with the previous frame.
                rectangle_center -= motion_estimate

                # Compute the ID of the rectangle in the global map which is most similar to the current rectangle.
                most_similar_ID = self.__find_most_similar_module(rectangle, area_threshold_ratio=0.5)

                if most_similar_ID is None:
                    correspondences.append(None)
                else:
                    # If the query rectangle's center is inside the nearest rectangle, set it as a correspondence.
                    closest_rectangle = self.global_module_map[most_similar_ID].last_rectangle
                    if rectangle_contains(closest_rectangle, rectangle_center):
                        correspondences.append(most_similar_ID)
                    else:
                        correspondences.append(None)

            # Update the current module map representation by considering the correspondences determined above.
            for rectangle_index, correspondence in enumerate(correspondences):
                # If there was no correspondence, add the rectangle as a new module in the global map.
                if correspondence is None:
                    next_ID = ID.next_id()
                    self.global_module_map[next_ID] = self.__ModuleInMap(next_ID, rectangle_list[rectangle_index],
                                                                         frame_id)
                else:
                    # Update the correspondet module in the map with the newest coordinates.
                    self.global_module_map[correspondence].add(rectangle_list[rectangle_index], frame_id)

            # Update the rectangles in the global map with the motion estimate.
            for _, rectangle_in_map in self.global_module_map.items():
                rectangle_in_map.add_motion(frame_id, motion_estimate)

        # All modules which have not been detected for more than a fixed timespan, are shifted to a database which
        # stores them but will not update their coordinates if an association is found.
        self.__store_old_modules(frame_id)

    def update_class_belief(self, probabilities: dict) -> None:
        """Updates the current class probability for the modules being detected in the last step.

        :param probabilities: A dictionary keyed by the module ID, whose value is a probability distribution over the classes.
        """
        for module_id, prob in probabilities.items():
            self.global_module_map[module_id].update_probability(prob)

    def __find_most_similar_module(self, rectangle: np.ndarray, area_threshold_ratio: float) -> int:
        """Finds the most similar rectangle in the global map by computing the surface between each rectangle stored in
        the module map, and the one passed as argument.

        :param rectangle: Query rectangle in the form np.array([[x0, y0], [x1, y1], [x2,y2], [x3, y3]]).
        :param area_threshold_ratio: The most similar module is accepted only if the relative deviation between the
        surface area of the query rectangle and the candidate is smaller than this parameter.
        :return: Index of the most similar rectangle in the module map.
        """
        rectangle_area = area(rectangle)
        rectangle_center = np.mean(rectangle, axis=0)
        min_surface_between_rect = np.infty
        best_id = None
        for module_id, module_in_map in self.global_module_map.items():
            if not rectangle_contains(module_in_map.last_rectangle, rectangle_center):
                continue
            surface_between_rect = area_between_rectangles(rectangle, module_in_map.last_rectangle)
            surface_diff = module_in_map.last_area - rectangle_area
            if surface_between_rect < min_surface_between_rect and surface_diff / rectangle_area < area_threshold_ratio:
                min_surface_between_rect = surface_between_rect
                best_id = module_id

        return best_id

    def __store_old_modules(self, current_frame_id: int) -> None:
        """Stores all modules in the global map which have not been observed for a fixed time span.

        :param current_frame_id: The current frame id.
        """
        old_rectangles_indices = []
        max_time_distance = 10
        for rect_id, rectangle_in_map in self.global_module_map.items():
            if current_frame_id - rectangle_in_map.frame_id_history[-1] > max_time_distance:
                old_rectangles_indices.append(rect_id)

        backup_rectangles = []
        for rectangle_id in old_rectangles_indices:
            rect = self.global_module_map[rectangle_id]
            backup_rectangles.append(rect)
            del self.global_module_map[rectangle_id]

        self.module_database.extend(backup_rectangles)
