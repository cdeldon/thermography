import tensorflow as tf
from abc import ABC, abstractmethod


class BaseNet(ABC):
    """
    Base interdace for nets used by the thermography package
    """

    def __init__(self, x: tf.placeholder, num_classes: int, name: str = "ThermoNet"):
        self.__x = x
        self.__num_classes = num_classes
        self.__name = name
        self.__logits = None

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x_: tf.placeholder):
        if type(x_) is not tf.placeholder:
            raise TypeError("__x in {} must be a tensorflow placeholder!".format(self.__class__.__name__))
        self.__x = x_

    @property
    def name(self):
        return self.__name

    @property
    def num_classes(self):
        if self.__logits is None:
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
    def logits(self, l: tf.placeholder):
        if type(l) is not tf.placeholder:
            raise TypeError("Logits in {} must be a tf.placeholder".format(self.__class__.__name__))
        self.__logits = l

    @abstractmethod
    def create(self) -> None:
        pass