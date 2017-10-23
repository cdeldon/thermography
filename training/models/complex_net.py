import tensorflow as tf
from .base_net import BaseNet
from .operations import *


class ComplexNet(BaseNet):
    def __init__(self, x: tf.placeholder, num_classes: int, keep_prob: float, *args, **kwargs):
        super(self.__class__, self).__init__(x=x, num_classes=num_classes, name="ComplexNet")
        self.keep_probability = keep_prob

        self.create()

    def create(self):
        with tf.variable_scope(self.name):
            with tf.variable_scope('conv_1'):
                h_conv1 = conv_relu(x=self.x, kernel_shape=[5, 5, 1, 5], bias_shape=[5])
                h_pool1 = max_pool_2x2(name="max_pool", x=h_conv1)
                # 12 15
            with tf.variable_scope('conv_2'):
                h_conv2 = conv_relu(x=h_pool1, kernel_shape=[5, 5, 5, 5], bias_shape=[5])
                h_pool2 = max_pool_2x2(name="max_pool", x=h_conv2)
                # 6 8

            with tf.variable_scope('conv_3'):
                h_conv3 = conv_relu(x=h_pool2, kernel_shape=[5, 5, 5, 7], bias_shape=[7])
                h_pool3 = max_pool_2x2(name="max_pool", x=h_conv3)
                # 3 4

            with tf.variable_scope('full_connected_1'):
                h_pool1_flat = tf.reshape(h_pool3, [-1, 3 * 4 * 7])

                W_fc1 = weight_variable(name="W", shape=[3 * 4 * 7, 32])
                b_fc1 = bias_variable(name="b", shape=[32])

                h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1 + b_fc1))

                with tf.variable_scope('drop_out_1'):
                    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=self.keep_probability, name="dropout")

            with tf.variable_scope('full_connected_2'):
                W_fc2 = weight_variable(name="W", shape=[32, 16])
                b_fc2 = bias_variable(name="b", shape=[16])

                h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2 + b_fc2))

                with tf.variable_scope('drop_out_2'):
                    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob=self.keep_probability, name="dropout")

            with tf.variable_scope('full_connected_3'):
                W_fc3 = weight_variable(name="W", shape=[16, self.num_classes])
                b_fc3 = bias_variable(name="b", shape=[self.num_classes])

                self.logits = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
