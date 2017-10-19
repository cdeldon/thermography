import tensorflow as tf


class SimpleNet(object):
    def __init__(self, x: tf.placeholder, keep_prob: float, num_classes: int):
        # Parse input arguments into class variables
        self.x = x
        self.y_conv = None
        self.num_classes = num_classes
        self.keep_probability = keep_prob

        self.create()

    def create(self):
        with tf.variable_scope('simple_model'):
            with tf.variable_scope('conv_1'):
                h_conv1 = conv_relu(x=self.x, kernel_shape=[5, 5, 1, 5], bias_shape=[5])
                h_pool1 = max_pool_2x2(name="max_pool", x=h_conv1)
                # 12 15
            with tf.variable_scope('conv_2'):
                h_conv2 = conv_relu(x=h_pool1, kernel_shape=[5, 5, 5, 5], bias_shape=[5])
                h_pool2 = max_pool_2x2(name="max_pool", x=h_conv2)
                # 6 8

            with tf.variable_scope('full_connected_1'):
                h_pool1_flat = tf.reshape(h_pool2, [-1, 6 * 8 * 5])

                W_fc1 = weight_variable(name="W", shape=[6 * 8 * 5, 64])
                b_fc1 = bias_variable(name="b", shape=[64])

                h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1 + b_fc1))

                with tf.variable_scope('drop_out_1'):
                    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=self.keep_probability, name="dropout")

            with tf.variable_scope('full_connected_2'):
                W_fc2 = weight_variable(name="W", shape=[64, 32])
                b_fc2 = bias_variable(name="b", shape=[32])

                h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2 + b_fc2))

                with tf.variable_scope('drop_out_2'):
                    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob=self.keep_probability, name="dropout")

            with tf.variable_scope('full_connected_3'):
                W_fc3 = weight_variable(name="W", shape=[32, self.num_classes])
                b_fc3 = bias_variable(name="b", shape=[self.num_classes])

                self.y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3


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
