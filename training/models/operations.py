import tensorflow as tf

__all__ = ["weight_variable",
           "bias_variable",
           "conv2d",
           "conv_relu",
           "max_pool_2x2",
           "max_pool_4x4"]


def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))


def bias_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(value=0.1))


def conv2d(name, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def conv_relu(x, kernel_shape, bias_shape):
    weights = weight_variable(name="W", shape=kernel_shape)
    biases = bias_variable(name="b", shape=bias_shape)
    return tf.nn.relu(conv2d(name="conv2d", x=x, W=weights) + biases)


def max_pool_2x2(name, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def max_pool_4x4(name, x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME', name=name)
