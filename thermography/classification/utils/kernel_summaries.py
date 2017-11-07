import tensorflow as tf


def kernel_to_image_summary(kernel: tf.Tensor, summary_name: str, max_images=3, collection: str = "kernels"):
    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    weights_0_to_1 = (kernel - x_min) / (x_max - x_min)

    weights_transposed = tf.transpose(weights_0_to_1, [3, 0, 1, 2])
    weights_transposed = tf.unstack(weights_transposed, axis=3)
    weights_transposed = tf.concat(weights_transposed, axis=0)
    weights_transposed = tf.expand_dims(weights_transposed, axis=-1)

    tf.summary.image(summary_name, weights_transposed, max_outputs=max_images, collections=[collection])
