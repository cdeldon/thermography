import numpy as np
import tensorflow as tf

from thermography.classification.utils.operations import *
from .base_net import BaseNet


class ThermoNet(BaseNet):
    def __init__(self, x: tf.Tensor, image_shape: np.ndarray, num_classes: int, keep_prob: float, *args, **kwargs):
        super(self.__class__, self).__init__(x=x, image_shape=image_shape, num_classes=num_classes, name="ThermoNet")
        self.keep_probability = keep_prob

        self.create()

    def create(self):
        with tf.variable_scope(self.name):
            current_shape = self.flat_shape
            with tf.variable_scope('conv_1'):
                h_conv1_0 = conv_relu(x=self.x, kernel_shape=[5, 5, self.image_shape[2], 8], bias_shape=[8], name="_0")
                self.h_pool1 = max_pool_4x4(name="max_pool", x=h_conv1_0)
                current_shape = self.update_shape(current_shape, 4)
                # 24 30

            with tf.variable_scope('conv_2'):
                h_conv2_0 = conv_relu(x=self.h_pool1, kernel_shape=[5, 5, 8, 16], bias_shape=[16], name="_0")
                self.h_pool2 = max_pool_4x4(name="max_pool", x=h_conv2_0)
                current_shape = self.update_shape(current_shape, 4)
                # 6 8

            with tf.variable_scope('full_connected_1'):
                flattened = tf.reshape(self.h_pool2, [-1, np.prod(current_shape) * 16])
                shape = flattened.get_shape().as_list()

                W_fc1 = weight_variable(name="W", shape=[shape[1], 256])
                b_fc1 = bias_variable(name="b", shape=[256])

                h_fc1 = tf.nn.relu(tf.matmul(flattened, W_fc1) + b_fc1)

                with tf.variable_scope('drop_out_1'):
                    self.h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=self.keep_probability, name="dropout")

            with tf.variable_scope('full_connected_2'):
                W_fc2 = weight_variable(name="W", shape=[256, self.num_classes])
                b_fc2 = bias_variable(name="b", shape=[self.num_classes])

                self.logits = tf.add(tf.matmul(self.h_fc1_drop, W_fc2), b_fc2, name="logits")
