"""
"""
import tensorflow as tf

from six.moves import range


def leaky_relu(x, leak=0.2, name="lrelu"):
    """
    https://github.com/tensorflow/tensorflow/issues/4079
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1.0 + leak)
        f2 = 0.5 * (1.0 - leak)
        return f1 * x + f2 * abs(x)


def discriminator(source, reuse):
    """
    build the discriminator network.
    source:
        The input image to discriminate
    reuse:
        Reuse the network. The network is used for traning both discriminator
        and generator. When training generator, use the loss from discriminator
        without updating it.
    """
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    # arXiv:1511.06434v2
    # build convolutional net to downsample.
    # no pooling layers.
    for layer_idx in range(4):
        # arXiv:1511.06434v2
        # in discriminator, use batch norm except input layer
        if layer_idx == 0:
            normalizer_fn = None
        else:
            normalizer_fn = tf.contrib.layers.batch_norm

        # arXiv:1511.06434v2
        # in discriminator, use LeakyReLU
        source = tf.contrib.layers.convolution2d(
            inputs=source,
            num_outputs=2 ** (7 + layer_idx),
            kernel_size=5,
            stride=2,
            padding='SAME',
            activation_fn=leaky_relu,
            normalizer_fn=normalizer_fn,
            weights_initializer=weights_initializer,
            scope='d_conv_{}'.format(layer_idx),
            reuse=reuse)

    return tf.contrib.layers.flatten(source)


def generator(seed):
    """
    build the generator network.
    """
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    # fully connected layer to upscale the seed for the input of
    # convolutional net.
    target = tf.contrib.layers.fully_connected(
        inputs=seed,
        num_outputs=4 * 4 * 1024,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        weights_initializer=weights_initializer,
        scope='g_project')

    # reshape to images
    target = tf.reshape(target, [-1, 4, 4, 1024])

    # transpose convolution to upscale
    for layer_idx in xrange(4):
        if layer_idx == 3:
            num_outputs = 3

            # arXiv:1511.06434v2
            # use tanh in output layer
            activation_fn = tf.nn.tanh

            # arXiv:1511.06434v2
            # use batch norm except the output layer
            normalizer_fn = None
        else:
            num_outputs = 2 ** (9 - layer_idx)

            # arXiv:1511.06434v2
            # use ReLU
            activation_fn = tf.nn.relu

            # arXiv:1511.06434v2
            # use batch norm
            normalizer_fn = tf.contrib.layers.batch_norm

        target = tf.contrib.layers.convolution2d_transpose(
            inputs=target,
            num_outputs=num_outputs,
            kernel_size=5,
            stride=2,
            padding='SAME',
            activation_fn=activation_fn,
            normalizer_fn=normalizer_fn,
            weights_initializer=weights_initializer,
            scope='g_conv_t_{}'.format(layer_idx))

    return target
