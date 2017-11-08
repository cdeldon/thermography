import tensorflow as tf

from ..utils import kernel_to_image_summary

__all__ = ["weight_variable",
           "bias_variable",
           "conv2d",
           "conv_relu",
           "max_pool_2x2",
           "max_pool_4x4",
           "max_pool_kxk"]


def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))


def bias_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(value=0.1))


def conv2d(name, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def conv_relu(x, kernel_shape, bias_shape, name: str = ""):
    weights = weight_variable(name="W" + name, shape=kernel_shape)
    kernel_to_image_summary(kernel=weights, summary_name="kernels")
    biases = bias_variable(name="b" + name, shape=bias_shape)
    return tf.nn.relu(conv2d(name="conv2d" + name, x=x, W=weights) + biases)


def max_pool_2x2(name, x):
    return max_pool_kxk(name=name, x=x, k=2)


def max_pool_4x4(name, x):
    return max_pool_kxk(name=name, x=x, k=4)


def max_pool_kxk(name, x, k: int):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
