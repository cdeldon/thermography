import numpy as np
import tensorflow as tf

from .base_net import BaseNet
from .operations import *


class ComplexNet(BaseNet):
    def __init__(self, x: tf.Tensor, image_shape: np.ndarray, num_classes: int, keep_prob: float, *args, **kwargs):
        super(self.__class__, self).__init__(x=x, image_shape=image_shape, num_classes=num_classes, name="ComplexNet")
        self.keep_probability = keep_prob

        self.create()

    def create(self):
        with tf.variable_scope(self.name):
            current_shape = self.flat_shape
            with tf.variable_scope('conv_1'):
                h_conv1_0 = conv_relu(x=self.x, kernel_shape=[5, 5, 1, 8], bias_shape=[8], name="_0")
                h_pool1 = max_pool_2x2(name="max_pool", x=h_conv1_0)
                current_shape = self.update_shape(current_shape, 2)
                # 48 60

                h_pool1_flat = max_pool_kxk("max_pool_flatten", h_pool1, k=8)
                shape = h_pool1_flat.get_shape().as_list()
                h_pool1_flat = tf.reshape(h_pool1_flat, [-1, shape[1] * shape[2] * shape[3]])  # None x 384

            with tf.variable_scope('conv_2'):
                h_conv2_0 = conv_relu(x=h_pool1, kernel_shape=[3, 3, 8, 8], bias_shape=[8], name="_0")
                h_pool2 = max_pool_4x4(name="max_pool", x=h_conv2_0)
                current_shape = self.update_shape(current_shape, 4)
                # 12 15

                h_pool2_flat = max_pool_2x2("max_pool_flatten", h_pool2)
                shape = h_pool2_flat.get_shape().as_list()
                h_pool2_flat = tf.reshape(h_pool2_flat, [-1, shape[1] * shape[2] * shape[3]])  # None x 384

            with tf.variable_scope('conv_3'):
                h_conv3_0 = conv_relu(x=h_pool2, kernel_shape=[3, 3, 8, 16], bias_shape=[16], name="_0")
                h_pool3 = max_pool_2x2(name="max_pool", x=h_conv3_0)
                current_shape = self.update_shape(current_shape, 2)
                # 6 8

                h_pool3_flat = h_pool3
                shape = h_pool3_flat.get_shape().as_list()
                h_pool3_flat = tf.reshape(h_pool3_flat, [-1, shape[1] * shape[2] * shape[3]])  # None x 768

            with tf.variable_scope('full_connected_1'):
                flattened = tf.concat(values=[h_pool1_flat, h_pool2_flat, h_pool3_flat], axis=1)  # None x (384 + 384 + 768)
                shape = flattened.get_shape().as_list()

                W_fc1 = weight_variable(name="W", shape=[shape[1], 256])
                b_fc1 = bias_variable(name="b", shape=[256])

                h_fc1 = tf.nn.relu(tf.matmul(flattened, W_fc1) + b_fc1)

                with tf.variable_scope('drop_out_1'):
                    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=self.keep_probability, name="dropout")

            with tf.variable_scope('full_connected_2'):
                W_fc2 = weight_variable(name="W", shape=[256, 32])
                b_fc2 = bias_variable(name="b", shape=[32])

                h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

                with tf.variable_scope('drop_out_2'):
                    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob=self.keep_probability, name="dropout")

            with tf.variable_scope('full_connected_3'):
                W_fc3 = weight_variable(name="W", shape=[32, self.num_classes])
                b_fc3 = bias_variable(name="b", shape=[self.num_classes])

                self.logits = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
