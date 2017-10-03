import json
import numpy as np
import os


class Modules:
    def __init__(self, module_path: str):
        """
        Loads the modules parameters into the object.
        :param module_path: Absolute path to the module file parameter.
        """

        self.module_path = module_path

        with open(self.module_path) as param_file:
            self.module_params = json.load(param_file)

    def __str__(self):
        return "Module dimensions: {},\n" \
               "Aspect ratio: {}".format(self.dimensions, self.aspect_ratio)

    @property
    def dimensions(self):
        return np.array(self.module_params["dimensions"])

    @property
    def aspect_ratio(self):
        return self.dimensions[0] / self.dimensions[1]

    @property
    def module_path(self):
        return self.__module_path

    @module_path.setter
    def module_path(self, path: str):
        if not os.path.exists(path):
            raise FileExistsError("Module config file {} not found".format(self.module_path))
        if not path.endswith("json"):
            raise ValueError("Can only parse '.json' files, passed module file is {}".format(path))
        self.__module_path = path
