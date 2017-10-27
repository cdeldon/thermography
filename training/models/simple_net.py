import numpy as np
import tensorflow as tf

from .base_net import BaseNet
from .operations import *


class SimpleNet(BaseNet):
    def __init__(self, x: tf.Tensor, image_shape: np.ndarray, num_classes: int, *args, **kwargs):
        super(self.__class__, self).__init__(x=x, image_shape=image_shape, num_classes=num_classes, name="SimpleNet")

        self.create()

    def create(self) -> None:
        with tf.variable_scope(self.name):
            current_shape = self.flat_shape
            with tf.variable_scope('conv_1'):
                h_pool0 = max_pool_4x4(name="max_pool", x=self.x)
                current_shape = self.update_shape(current_shape, 4)

                h_conv1 = conv_relu(x=h_pool0, kernel_shape=[5, 5, self.channels, 3], bias_shape=[3])
                h_pool1 = max_pool_2x2(name="max_pool", x=h_conv1)
                current_shape = self.update_shape(current_shape, 2)
                # 12 15

            with tf.variable_scope('full_connected_1'):
                h_pool1_flat = tf.reshape(h_pool1, [-1, np.prod(current_shape) * 3])

                W_fc1 = weight_variable(name="W", shape=[np.prod(current_shape) * 3, self.num_classes])
                b_fc1 = bias_variable(name="b", shape=[self.num_classes])

                self.logits = tf.add(tf.matmul(h_pool1_flat, W_fc1), b_fc1)
