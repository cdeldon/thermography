import tensorflow as tf
from .base_net import BaseNet
from .operations import *


class SimpleNet(BaseNet):
    def __init__(self, x: tf.placeholder, num_classes: int, *args, **kwargs):
        super(self.__class__, self).__init__(x=x, num_classes=num_classes, name="SimpleNet")

        self.create()

    def create(self) -> None:
        with tf.variable_scope(self.name):
            with tf.variable_scope('conv_1'):
                self.h_pool0 = max_pool_4x4(name="max_pool", x=self.x)
                self.h_conv1 = conv_relu(x=self.h_pool0, kernel_shape=[5, 5, 1, 3], bias_shape=[3])
                self.h_pool1 = max_pool_2x2(name="max_pool", x=self.h_conv1)
                # 3 4

            with tf.variable_scope('full_connected_1'):
                self.h_pool1_flat = tf.reshape(self.h_pool1, [-1, 3 * 4 * 3])

                self.W_fc1 = weight_variable(name="W", shape=[3 * 4 * 3, self.num_classes])
                self.b_fc1 = bias_variable(name="b", shape=[self.num_classes])

                self.logits = tf.add(tf.matmul(self.h_pool1_flat, self.W_fc1), self.b_fc1)
