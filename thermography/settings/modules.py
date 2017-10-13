import numpy as np


class Modules:
    def __init__(self):
        """
        Initializes the module object with default parameters.
        """

        self.dimensions = np.array([1.5, 1.0])

    def __str__(self):
        return "Module dimensions: {},\n" \
               "Aspect ratio: {}".format(self.dimensions, self.aspect_ratio)

    @property
    def aspect_ratio(self):
        return self.dimensions[0] / self.dimensions[1]
