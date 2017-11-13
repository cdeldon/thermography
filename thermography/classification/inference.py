import os

import cv2
import numpy as np
import tensorflow as tf
from simple_logger import Logger

from .models.base_net import BaseNet


class Inference:
    """Class implementing the inference procedure to classify new images according to a preexisting moduel.

    :Example:

    .. code-block:: python

        classification_model = ThermoNet3x3
        inference = Inference(checkpoint_dir, classification_model, img_shape, num_classes)
        img_list = [img_1, img_2, ...] # len(img_list) = N
        class_probabilities = inference.classify(img_list) # class_probabilities.shape = [N, num_classes]

    """
    def __init__(self, checkpoint_dir: str, model_class: type, image_shape: np.ndarray, num_classes: int):
        """Initializes the inference object by loading the weights of an already trained model.

        :param checkpoint_dir: Directory of the checkpoint where the weights of the model are stored.
        :param model_class: Class associated to the stored weights. This class is used to build the :attr:`tf.graph` which will be used for the inference.
        :param image_shape: Shape of the images fed to the classifier. The images passed to the :func:`self.classify <thermography.classification.inference.Inference.classify>` function are resized according to this parameter.
        :param num_classes: Number of classes predicted by the model.

        .. warning:: The parameters must all be consistent with the preexisting weights!
        """
        self.checkpoint_dir = checkpoint_dir
        self.image_shape = image_shape
        self.num_classes = num_classes

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, *self.image_shape], name="input_image")
            self.keep_probability = tf.placeholder(tf.float32, name="keep_probability")
            self.model = model_class(x=self.x, image_shape=self.image_shape, num_classes=self.num_classes,
                                     keep_prob=self.keep_probability)

        self.logits = self.model.logits
        self.probabilities = tf.nn.softmax(self.logits)

        # Add ops to save and restore all the variables.
        self.sess = tf.Session(graph=self.graph)

        # Restore variables from disk.
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, self.model.name))

        Logger.info("Model restored.")

    def __del__(self):
        Logger.info("Deleting inference object")
        self.sess.close()

    @property
    def model(self):
        """Returns the model being used for classification."""
        return self.__model

    @model.setter
    def model(self, m: BaseNet):
        if not isinstance(m, BaseNet):
            raise TypeError("Model passed to {} is not deriving from BaseNet".format(self.__class__.__name__))
        self.__model = m

    def classify(self, image_list: list) -> np.ndarray:
        """Classifies the image list passed as argument using the model loaded in :attr:`self.model`.

        :param image_list: Python list of numpy arrays representing the images to be classified. All images are classified as a mini-batch.
        :return: A numpy array of shape `[len(image_list), self.num_classes]` containing the class probability for each image passed as argument.

        .. note:: If the images contained in the input parameter are not of the same shape as the one store in :attr:`self.image_shape`, the input images are resized to fit the desired image shape.
        """
        if len(image_list) == 0:
            return np.empty(shape=[0])

        img_tensor = []
        for img in image_list:
            if (img.shape[0:2] != self.image_shape[0:2]).any():
                shape = img.shape
                img = img.astype(np.float32)
                Logger.warning("Image is of size {}, should be {},  resizing".format(shape, self.image_shape))
                img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_AREA)
            if img.shape[2] != self.image_shape[2]:
                if self.image_shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img
                elif self.image_shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img_mean = np.mean(img, axis=(0, 1))
            img_std = np.std(img, axis=(0, 1))
            img = (img - img_mean) / img_std

            img_tensor.append(img)

        img_tensor = np.array(img_tensor)

        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor[..., np.newaxis]

        Logger.debug("Classifying {} module image{}".format(
            img_tensor.shape[0], "" if img_tensor.shape[0] == 1 else "s"))

        class_probabilities = self.sess.run(self.probabilities,
                                            feed_dict={self.x: img_tensor, self.keep_probability: 1.0})
        return class_probabilities
