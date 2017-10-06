import cv2
import numpy as np
from thermography.utils import ID, rectangle_contains


class ModuleMap:
    class __RectangleInMap:
        def __init__(self, rectangle: np.ndarray, frame_id: int):
            self.last_rectangle = None
            self.last_center = None

            self.frame_id_history = []
            self.rectangle_history = {}

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

    def __init__(self):
        # A dictionary of rectangles and their centers keyed by their ID.
        self.global_rectangle_map = {}

    def __repr__(self):
        s = ""
        for rectangle_id, rectangle_in_map in self.global_rectangle_map.items():
            s += str(rectangle_in_map) + "\n\n"
        return s

    def insert(self, rectangle_list: list, frame_id: int, motion_estimate: np.ndarray = None):

        if motion_estimate is None:
            motion_estimate = np.array([0.0, 0.0])

        # In case there are no rectangles in the map, store them all.
        if len(self.global_rectangle_map) == 0:
            for rectangle in rectangle_list:
                self.global_rectangle_map[ID.next_id()] = self.__RectangleInMap(rectangle, frame_id)
        else:
            associations = {}
            for rectangle_index, rectangle in enumerate(rectangle_list):
                rectangle_center = np.mean(rectangle, axis=0)
                # Shift the rectangle center using the motion estimate.
                rectangle_center -= motion_estimate

                match_index = self.__find_closest_rectangle(rectangle_center)
                if match_index is not None:
                    # If this rectangle's center is inside the matched rectangle, add it as a correspondence.
                    closest_rectangle = self.global_rectangle_map[match_index].last_rectangle
                    if rectangle_contains(closest_rectangle, rectangle_center):
                        associations[rectangle_index] = match_index
                    else:
                        associations[rectangle_index] = None
                else:
                    associations[rectangle_index] = None

            for rectangle_index, correspondence in associations.items():
                if correspondence is None:
                    self.global_rectangle_map[ID.next_id()] = self.__RectangleInMap(rectangle_list[rectangle_index],
                                                                                    frame_id)
                else:
                    self.global_rectangle_map[correspondence].add(rectangle_list[rectangle_index], frame_id)

    def __find_closest_rectangle(self, rectangle_center: np.ndarray) -> int:
        min_distance = np.infty
        best_id = None
        for rect_id, rectangle_in_map in self.global_rectangle_map.items():
            dist = np.linalg.norm(rectangle_center - rectangle_in_map.last_center)
            if dist < min_distance:
                min_distance = dist
                best_id = rect_id

        return best_id
