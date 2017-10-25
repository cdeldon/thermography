import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod


class BaseNet(ABC):
    """
    Base interdace for nets used by the thermography package
    """

    def __init__(self, x: tf.Tensor, image_shape: list, num_classes: int, name: str = "ThermoNet"):
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
        if type(shape) is not list:
            raise TypeError("__image_shape in {} must be a list of two elements".format(self.__class__.__name__))
        if len(shape) != 2:
            raise ValueError("__image_shape in {} must be a list of two elements".format(self.__class__.__name__))
        self.__image_shape = shape

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
        current_shape = current_shape.astype(np.float32) / scale
        return np.ceil(current_shape).astype(np.int32)
