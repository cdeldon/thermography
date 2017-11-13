import tensorflow as tf


def kernel_to_histogram_summary(kernel: tf.Tensor, summary_name: str, collection: str = "histograms") -> None:
    """Generates a summary histogram from the kernel passed as argument and adds it to the collection specified.

    :param kernel: Tensor representing a kernel.
    :param summary_name: Name to give to the generated summary.
    :param collection: Summary collection where the histogram summary is added.
    """
    tf.summary.histogram(name=summary_name, values=kernel, collections=[collection])


def kernel_to_image_summary(kernel: tf.Tensor, summary_name: str, max_images=3, collection: str = "kernels") -> None:
    """
    Converts a kernel tensor of shape [width, height, in_channels, out_channels] to an image summary.

    :param kernel: Tensor representing the convolutional kernel.
    :param summary_name: Name to give to the summary.
    :param max_images: Maximal number of images to extract from the kernel tensor (slices).
    :param collection: Summary collection where the image summary is added.
    """
    # Normalize the input kernel to the 0-1 range.
    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    weights_0_to_1 = (kernel - x_min) / (x_max - x_min)

    # Rearrange weights such that they are ordered as [out_channels, width, height, in_channels]
    weights_transposed = tf.transpose(weights_0_to_1, [3, 0, 1, 2])
    # Unstack the in_channels axis --> [0, [out_channels, width, height], 1: [...], ..., in_channels-1: [...]]
    weights_transposed = tf.unstack(weights_transposed, axis=3)
    # Concatenate the unstacked channels: --> [out_channels * in_channels, width, height]
    weights_transposed = tf.concat(weights_transposed, axis=0)
    # Add an empty dimension at the end of the tensor [out_channels * in_channels, width, height, 1]
    weights_transposed = tf.expand_dims(weights_transposed, axis=-1)

    tf.summary.image(summary_name, weights_transposed, max_outputs=max_images, collections=[collection])
