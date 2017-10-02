import cv2
import json
import numpy as np
import os


class Camera:
    def __init__(self, camera_path: str):
        """
        Loads the camera parameters into the object.
        :param camera_path: Absolute path to the camera file parameter.
        """

        self.camera_path = camera_path

        with open(self.camera_path) as param_file:
            self.camera_params = json.load(param_file)

    def __str__(self):
        return "Image size: {},\n" \
               "Focal length: {}\n" \
               "Principal point: {}\n" \
               "Radial distortion: {}, {}, {}\n" \
               "Tangential distortion: {}, {}".format(self.image_size, self.focal_length, self.principal_point, self.r1,
                                                      self.r2, self.r3, self.t1, self.t2)

    @property
    def camera_matrix(self):
        return np.array([np.array([self.focal_length, 0, self.principal_point[0]]),
                         np.array([0, self.focal_length, self.principal_point[1]]),
                         np.array([0, 0, 1])])

    @property
    def distortion_coeff(self):
        return np.array([self.r1, self.r2, self.t1, self.t2, self.r3])

    @property
    def image_size(self):
        return np.array(self.camera_params["image_size"])

    @property
    def focal_length(self):
        return self.camera_params["focal_length"]

    @property
    def principal_point(self):
        return np.array(self.camera_params["principal_point"])

    @property
    def r1(self):
        return self.camera_params["distortion"]["radial"]["r1"]

    @property
    def r2(self):
        return self.camera_params["distortion"]["radial"]["r2"]

    @property
    def r3(self):
        return self.camera_params["distortion"]["radial"]["r3"]

    @property
    def t1(self):
        return self.camera_params["distortion"]["tangential"]["t1"]

    @property
    def t2(self):
        return self.camera_params["distortion"]["tangential"]["t2"]

    @property
    def camera_path(self):
        return self.__camera_path

    @camera_path.setter
    def camera_path(self, path: str):
        if not os.path.exists(path):
            raise FileExistsError("Camera config file {} not found".format(self.camera_path))
        if not path.endswith("json"):
            raise ValueError("Can only parse '.json' files, passed camera file is {}".format(path))
        self.__camera_path = path
