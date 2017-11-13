import numpy as np
import tensorflow as tf

from thermography.classification.utils.operations import *
from .base_net import BaseNet


class ThermoNet3x3(BaseNet):
    """Class representing the computational graph structure used for classification of solar panel modules."""

    def __init__(self, x: tf.Tensor, image_shape: np.ndarray, num_classes: int, keep_prob: float, *args, **kwargs):
        """Initializes the computational graph according to the structure defined in :func:`self.create <.ThermoNet3x3.create>`.

        :param x: Tensorflow placeholder which will be fed with a batch of images.
        :param image_shape: Shape of each input image fed to :attr:`self.x`.
        :param num_classes: Number of classes which the model can predict.
        :param keep_prob: Tensorflow placeholder representing the keep probability for the dropout layers.
        """
        super(self.__class__, self).__init__(x=x, image_shape=image_shape, num_classes=num_classes, name="ThermoNet3x3")
        self.keep_probability = keep_prob

        self.create()

    def create(self):
        """Creates the ThermoNet3x3 computation graph consisting of three convolutional layers followed by three fully connected layers."""
        with tf.variable_scope(self.name):
            current_shape = self.flat_shape
            with tf.variable_scope('conv_1'):
                h_conv1_0 = conv_relu(x=self.x, kernel_shape=[5, 5, self.image_shape[2], 8], bias_shape=[8], name="_0")
                self.h_pool1 = max_pool_2x2(name="max_pool", x=h_conv1_0)
                current_shape = self.update_shape(current_shape, 2)
                # 48 60

            with tf.variable_scope('conv_2'):
                h_conv2_0 = conv_relu(x=self.h_pool1, kernel_shape=[3, 3, 8, 16], bias_shape=[16], name="_0")
                self.h_pool2 = max_pool_4x4(name="max_pool", x=h_conv2_0)
                current_shape = self.update_shape(current_shape, 4)
                # 12 15

            with tf.variable_scope('conv_3'):
                h_conv3_0 = conv_relu(x=self.h_pool2, kernel_shape=[3, 3, 16, 32], bias_shape=[32], name="_0")
                self.h_pool3 = max_pool_2x2(name="max_pool", x=h_conv3_0)
                current_shape = self.update_shape(current_shape, 2)
                # 6 8

            with tf.variable_scope('drop_out_1'):
                self.h_pool3_drop = tf.nn.dropout(self.h_pool3, keep_prob=self.keep_probability, name="dropout")

            flattened = tf.reshape(self.h_pool3_drop, [-1, np.prod(current_shape) * 32], name="flattened")

            with tf.variable_scope('fc_1'):
                shape = flattened.get_shape().as_list()

                W_fc1 = weight_variable(name="W", shape=[shape[1], 256])
                b_fc1 = bias_variable(name="b", shape=[256])

                self.h_fc1 = tf.nn.relu(tf.matmul(flattened, W_fc1) + b_fc1)

            with tf.variable_scope('fc_2'):
                W_fc2 = weight_variable(name="W", shape=[256, 32])
                b_fc2 = bias_variable(name="b", shape=[32])

                h_fc2 = tf.nn.relu(tf.matmul(self.h_fc1, W_fc2) + b_fc2)

            with tf.variable_scope('drop_out_2'):
                self.h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob=self.keep_probability, name="dropout")

            with tf.variable_scope('readout'):
                W_fc3 = weight_variable(name="W", shape=[32, self.num_classes])
                b_fc3 = bias_variable(name="b", shape=[self.num_classes])

                self.logits = tf.add(tf.matmul(self.h_fc2_drop, W_fc3), b_fc3, name="logits")
