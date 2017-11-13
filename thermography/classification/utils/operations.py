"""Module containing utility functions to combine multiple tensorflow operations into a single function call."""

import tensorflow as tf

from . import kernel_to_image_summary, kernel_to_histogram_summary

__all__ = ["weight_variable",
           "bias_variable",
           "conv2d",
           "conv_relu",
           "max_pool_2x2",
           "max_pool_4x4",
           "max_pool_kxk"]


def weight_variable(name: str, shape: list) -> tf.Tensor:
    """Generates or returns an existing tensorflow variable to be used as weight.

    :param name: Name of the variable to be returned.
    :param shape: Shape of the variable to be returned.
    :return A tf.Variable with the name and shape passed as argument, initialized with a truncated normal initializer.
    """
    return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))


def bias_variable(name: str, shape: list) -> tf.Tensor:
    """Generates or returns an existing tensorflow variable to be used as a bias.

    :param name: Name of the variable to be returned.
    :param shape: Shape of the variable to be returned.
    :return: A tf.variable with the name and shape passed as argument, initialized with a constant initializer.
    """
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(value=0.1))


def conv2d(name: str, x: tf.Tensor, W: tf.Tensor) -> tf.Tensor:
    """Returns the graph node associated to the convolution between the input parameters.

    :param name: Name to give to the resulting convolution.
    :param x: Tensor to be convolved.
    :param W: Weight variable to be used in the convolution.
    :return: A new tensor consisting of the convolution between the two input parameters.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)


def conv_relu(x: tf.Tensor, kernel_shape: list, bias_shape: list, name: str = "") -> tf.Tensor:
    """Performs and returns a convolution, followed by bias addition and non-linearity (relu).

    :param x: Input tensor to the convolution-bias-relu operation.
    :param kernel_shape: Kernel shape to be used in the convolution.
    :param bias_shape: Bias shape to be added to the result of the convolution.
    :param name: Name of the returned operation.
    :return: A new tensor consisting of the convolution-bias-relu operation.
    """
    weights = weight_variable(name="W" + name, shape=kernel_shape)
    kernel_to_histogram_summary(kernel=weights, summary_name="W")
    kernel_to_image_summary(kernel=weights, summary_name="kernels")
    biases = bias_variable(name="b" + name, shape=bias_shape)
    kernel_to_histogram_summary(kernel=biases, summary_name="b")
    return tf.nn.relu(conv2d(name="conv2d" + name, x=x, W=weights) + biases)


def max_pool_2x2(name: str, x: tf.Tensor) -> tf.Tensor:
    """Performs a max_pool of size 2x2 to the input parameter.

    :param name: Name to assign to the returned operation.
    :param x: input tensor.
    """
    return max_pool_kxk(name=name, x=x, k=2)


def max_pool_4x4(name: str, x: tf.Tensor) -> tf.Tensor:
    """Performs a max_pool of size 4x4 to the input parameter.

    :param name: Name to assign to the returned operation.
    :param x: input tensor.
    """
    return max_pool_kxk(name=name, x=x, k=4)


def max_pool_kxk(name: str, x: tf.Tensor, k: int) -> tf.Tensor:
    """Performs a max_pool of size kxk to the input parameter.

    :param name: Name to assign to the returned operation.
    :param x: input tensor.
    :param k: size of the pooling operation.
    """
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)
