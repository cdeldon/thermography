import json
import os

import numpy as np
from simple_logger import Logger


class Camera:
    """Class representing the intrinsic camera parameters of the camera used to capture the videos analyzed by :mod:`thermography`."""

    def __init__(self, camera_path: str):
        """Loads the camera parameters into the object.

        :param camera_path: Absolute path to the camera file parameter.
        """

        self.camera_path = camera_path

        with open(self.camera_path) as param_file:
            self.camera_params = json.load(param_file)

        Logger.debug("Camera parameter file is: \n{}".format(str(self)))

    def __str__(self):
        return "Image size: {},\n" \
               "Focal length: {}\n" \
               "Principal point: {}\n" \
               "Radial distortion: {}, {}, {}\n" \
               "Tangential distortion: {}, {}".format(self.image_size, self.focal_length, self.principal_point, self.r1,
                                                      self.r2, self.r3, self.t1, self.t2)

    @property
    def camera_matrix(self) -> np.ndarray:
        """Returns the intrinsic camera matrix."""
        return np.array([np.array([self.focal_length, 0, self.principal_point[0]]),
                         np.array([0, self.focal_length, self.principal_point[1]]),
                         np.array([0, 0, 1])])

    @property
    def distortion_coeff(self) -> np.ndarray:
        """Returns the distortion coefficients of the camera."""
        return np.array([self.r1, self.r2, self.t1, self.t2, self.r3])

    @property
    def image_size(self) -> np.ndarray:
        """Returns the image size captured by the camera."""
        return np.array(self.camera_params["image_size"])

    @property
    def focal_length(self) -> float:
        """Returns the focal length of the camera expressed in pixel units."""
        return self.camera_params["focal_length"]

    @property
    def principal_point(self) -> np.ndarray:
        """Returns the pixel coordinates of the principal point."""
        return np.array(self.camera_params["principal_point"])

    @property
    def r1(self) -> float:
        "Returns the first radial distortion coefficient."
        return self.camera_params["distortion"]["radial"]["r1"]

    @property
    def r2(self) -> float:
        """Returns the second radial distortion coefficient."""
        return self.camera_params["distortion"]["radial"]["r2"]

    @property
    def r3(self) -> float:
        """Returns the thirds radial distortion coefficient."""
        return self.camera_params["distortion"]["radial"]["r3"]

    @property
    def t1(self) -> float:
        """Returns the first tangential distortion coefficient."""
        return self.camera_params["distortion"]["tangential"]["t1"]

    @property
    def t2(self) -> float:
        """Returns the second tangential distortion coefficient."""
        return self.camera_params["distortion"]["tangential"]["t2"]

    @property
    def camera_path(self) -> str:
        """Returns the absolute path to the configuarion file associated to the camera parameters contained in this object."""
        return self.__camera_path

    @camera_path.setter
    def camera_path(self, path: str):
        if not os.path.exists(path):
            Logger.fatal("Camera config file {} not found".format(self.camera_path))
            raise FileNotFoundError("Camera config file {} not found".format(self.camera_path))
        if not path.endswith("json"):
            Logger.fatal("Can only parse '.json' files")
            raise ValueError("Can only parse '.json' files, passed camera file is {}".format(path))
        self.__camera_path = path
