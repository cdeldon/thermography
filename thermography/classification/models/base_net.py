from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class BaseNet(ABC):
    """Base interface for nets used by the :mod:`thermography` package. This class offers a convenience interface to train a classifier and to access its data."""

    def __init__(self, x: tf.Tensor, image_shape: np.ndarray, num_classes: int, name: str = "ThermoNet"):
        self.x = x
        self.image_shape = image_shape
        self.__num_classes = num_classes
        self.__name = name
        self.__logits = None

    @property
    def x(self) -> tf.Tensor:
        """Returns a reference to the input placeholder."""
        return self.__x

    @x.setter
    def x(self, x_: tf.Tensor):
        if type(x_) is not tf.Tensor:
            raise TypeError("__x in {} must be a tensorflow placeholder!".format(self.__class__.__name__))
        self.__x = x_

    @property
    def image_shape(self) -> np.ndarray:
        """Returns the image shape accepted by the model. This shape corresponds to the shape of each element carried by the input placeholder :attr:`self.x`."""
        return self.__image_shape

    @image_shape.setter
    def image_shape(self, shape):
        if type(shape) is not np.ndarray:
            raise TypeError(
                "__image_shape in {} must be a  np.ndarray of three elements".format(self.__class__.__name__))
        if len(shape) != 3:
            raise ValueError(
                "__image_shape in {} must be a  np.ndarray of there elements".format(self.__class__.__name__))
        self.__image_shape = shape

    @property
    def channels(self)->int:
        """Returns the number of channels accepted by the model."""
        return self.image_shape[2]

    @property
    def name(self)->str:
        """Returns the name associated to the model."""
        return self.__name

    @property
    def num_classes(self) ->int:
        """Returns the number of classes which the model will consider for classification."""
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
    def logits(self) -> tf.Tensor:
        """Returns a tf.Tensor representing the logits of the classified input images."""
        if self.__logits is None:
            raise RuntimeError("__logits in {} is has not been overridden!".format(self.__class__.__name__))
        return self.__logits

    @logits.setter
    def logits(self, l: tf.Tensor):
        self.__logits = l

    @abstractmethod
    def create(self) -> None:
        """Method which each subclass must implement which creates the computational graph.

        .. note:: This method must use the input placeholder :attr:`self.x` and terminate into the logits tensor :attr:`self.logits`."""
        pass

    @staticmethod
    def update_shape(current_shape: np.ndarray, scale: int) -> np.ndarray:
        """Updates the current shape of the data in the computational graph when a pooling operation is applied to it.

        :Example:

            .. code-block:: python

                current_shape = self.flat_shape # [w, h]
                conv = conv_relu(x=self.x, kernel_shape, bias_shape)
                pool = max_pool_2x2(name="max_pool", x=conv)
                current_shape = self.update_shape(current_shape, 2) # [w/2, h/2]
        """
        assert (len(current_shape) == 2)
        return (np.ceil(current_shape.astype(np.float32) / scale)).astype(np.int32)

    @property
    def flat_shape(self) -> np.ndarray:
        """Returns the shape of the image which the model takes as input to the computational graph.

        :Example:

            .. code-block:: python

                img_batch.shape = [batch_shape, w, h, c]
                flat = model.flat_shape # [w, h]
        """
        return self.image_shape[0:2]
