from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class BaseNet(ABC):
    """
    Base interdace for nets used by the thermography package
    """

    def __init__(self, x: tf.Tensor, image_shape:  np.ndarray, num_classes: int, name: str = "ThermoNet"):
        self.x = x
        self.image_shape = image_shape
        self.__num_classes = num_classes
        self.__name = name
        self.__logits = None

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x_: tf.Tensor):
        if type(x_) is not tf.Tensor:
            raise TypeError("__x in {} must be a tensorflow placeholder!".format(self.__class__.__name__))
        self.__x = x_

    @property
    def image_shape(self):
        return self.__image_shape

    @image_shape.setter
    def image_shape(self, shape):
        if type(shape) is not  np.ndarray:
            raise TypeError("__image_shape in {} must be a  np.ndarray of three elements".format(self.__class__.__name__))
        if len(shape) != 3:
            raise ValueError("__image_shape in {} must be a  np.ndarray of there elements".format(self.__class__.__name__))
        self.__image_shape = shape

    @property
    def channels(self):
        return self.image_shape[2]

    @property
    def name(self):
        return self.__name

    @property
    def num_classes(self):
        if self.__num_classes is None:
            raise RuntimeError("__num_classes in {} is has not been overridden!".format(self.__class__.__name__))
        return self.__num_classes

    @num_classes.setter
    def num_classes(self, n: int):
        if type(n) is not int:
            raise TypeError("Num classes in {} must be an integer!".format(self.__class__.__name__))
        if n <= 0:
            raise ValueError("Num classes in {} must be strictly positive!".format(self.__class__.__name__))
        self.__num_classes = n

    @property
    def logits(self):
        if self.__logits is None:
            raise RuntimeError("__logits in {} is has not been overridden!".format(self.__class__.__name__))
        return self.__logits

    @logits.setter
    def logits(self, l: tf.Tensor):
        self.__logits = l

    @abstractmethod
    def create(self) -> None:
        pass

    @staticmethod
    def update_shape(current_shape: np.ndarray, scale: int):
        assert(len(current_shape) == 2)
        return (np.ceil(current_shape.astype(np.float32) / scale)).astype(np.int32)

    @property
    def flat_shape(self):
        return self.image_shape[0:2]
