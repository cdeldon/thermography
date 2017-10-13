import cv2
import numpy as np
from thermography.utils import ID, rectangle_contains


class ModuleMap:
    class __ModuleInMap:
        def __init__(self, ID: int, rectangle: np.ndarray, frame_id: int):
            self.ID = ID
            self.last_rectangle = None
            self.last_center = None

            self.frame_id_history = []
            self.rectangle_history = {}

            self.cumulated_motion = np.array([0, 0], dtype=np.float32)

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

        def add(self, rectangle: np.ndarray, frame_id: int):
            self.last_rectangle = rectangle
            self.last_center = np.mean(self.last_rectangle, axis=0)

            self.frame_id_history.append(frame_id)
            self.rectangle_history[frame_id] = rectangle

            self.cumulated_motion = np.array([0, 0], dtype=np.float32)

        def add_motion(self, frame_id: int, motion_estimate: np.ndarray):
            if frame_id != self.frame_id_history[-1]:
                self.cumulated_motion += motion_estimate

    def __init__(self):
        # A dictionary of modules and their centers keyed by their ID.
        self.global_module_map = {}
        self.module_database = []

    def __repr__(self):
        s = ""
        for module_id, module_in_map in self.global_module_map.items():
            s += str(module_in_map) + "\n\n"
        return s

    def insert(self, rectangle_list: list, frame_id: int, motion_estimate: np.ndarray = None):
        """
        Inserts the rectangles passed as first parameter into the global map representation of the module map.

        :param rectangle_list: List of detected rectangles to be inserted into the global module map.
        :param frame_id: ID of the current image frame associated to the detected rectangles.
        :param motion_estimate: Motion estimate between the last frame (ID-1) and the frame containing the rectangles.
        """

        # When no information about the motion estimate is given, assume no motion.
        if motion_estimate is None:
            motion_estimate = np.array([0.0, 0.0])

        # In case there are no rectangles in the global map, store all the ones passed to the function.
        if len(self.global_module_map) == 0:
            for rectangle in rectangle_list:
                # Give a new ID to each rectangle.
                next_ID = ID.next_id()
                self.global_module_map[next_ID] = self.__ModuleInMap(next_ID, rectangle, frame_id)
        else:
            # Associations between the rectangles passed to the function and the ones already stored in the global map.
            # If no association is found (e.g. a rectangle is new), then set the association to None.
            associations = []
            for rectangle in rectangle_list:
                rectangle_center = np.mean(rectangle, axis=0)
                # Shift the rectangle center using the motion estimate.
                rectangle_center -= motion_estimate

                # Compute the ID of the rectangle in the global map which is closest to the current rectangle.
                nearest_ID = self.__find_closest_module(rectangle_center)

                # If this rectangle's center is inside the nearest rectangle, set it as a correspondence.
                closest_rectangle = self.global_module_map[nearest_ID].last_rectangle
                if rectangle_contains(closest_rectangle, rectangle_center):
                    associations.append(nearest_ID)
                else:
                    associations.append(None)

            for rectangle_index, correspondence in enumerate(associations):
                if correspondence is None:
                    next_ID = ID.next_id()
                    self.global_module_map[next_ID] = self.__ModuleInMap(next_ID, rectangle_list[rectangle_index],
                                                                         frame_id)
                else:
                    self.global_module_map[correspondence].add(rectangle_list[rectangle_index], frame_id)

            for rectangle_index, rectangle_in_map in self.global_module_map.items():
                rectangle_in_map.add_motion(frame_id, motion_estimate)

        self.__store_old_modules(frame_id)

    def __find_closest_module(self, rectangle_center: np.ndarray) -> int:
        min_distance = np.infty
        best_id = None
        for module_id, module_in_map in self.global_module_map.items():
            dist = np.linalg.norm(rectangle_center - (module_in_map.last_center - module_in_map.cumulated_motion))
            if dist < min_distance:
                min_distance = dist
                best_id = module_id

        return best_id

    def __store_old_modules(self, current_frame_id : int):
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
