"""
"""
import tensorflow as tf


def build_discriminator(source_images, target_images, reuse=False):
    """
    arXiv:1611.07004v1
    build PatchGAN discriminator
    """
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    flow = tf.concat([source_images, target_images], axis=3)

    # arXiv:1611.07004v1
    # 5.1.2
    # 70 x 70 discriminator
    ks = [64, 128, 256, 512]

    for i, k in enumerate(ks):
        # arXiv:1611.07004v1
        # we achieve this variation in patch size by adjusting the depth of the
        # GAN discriminator.
        # flow = tf.pad(flow, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')

        # arXiv:1611.07004v1
        # 5.1.2
        # As an exception to the above notation, BatchNorm is not applied to
        # the first C64 layer. All ReLU are leaky, with slope 0.2.
        flow = tf.contrib.layers.convolution2d(
            inputs=flow,
            num_outputs=k,
            kernel_size=4,
            stride=2,
            padding='SAME',
            activation_fn=tf.contrib.keras.layers.LeakyReLU(alpha=0.2),
            normalizer_fn=None if i == 0 else tf.contrib.layers.batch_norm,
            weights_initializer=weights_initializer,
            scope='d_c_{}_{}'.format(i, k),
            reuse=reuse)

    # arXiv:1611.07004v1
    # 5.1.2
    # after the last layer, a convolution is applied to map to a 1 dimensional
    # output, followed by a Sigmoid function.
    flow = tf.contrib.layers.convolution2d(
        inputs=flow,
        num_outputs=1,
        kernel_size=4,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.sigmoid,
        weights_initializer=weights_initializer,
        scope='d_out',
        reuse=reuse)

    return flow


def build_generator(source_images, is_training):
    """
    arXiv:1611.07004v1
    build encoder-decoder generator
    """
    flow = source_images

    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    # arXiv:1611.07004v1
    # 5.1.1
    # encoder-decoder depth.
    ks = [64, 128, 256, 512, 512, 512, 512, 512]

    encoders = []

    print flow.shape

    # encoder
    for i, k in enumerate(ks):
        # arXiv:1611.07004v1
        # 5.1.1
        # BatchNorm is not applied to the first C64 layer in the encoder.
        # All ReLUs in the encoder are leaky, with slope 0.2.
        flow = tf.contrib.layers.convolution2d(
            inputs=flow,
            num_outputs=k,
            kernel_size=4,
            stride=2,
            padding='SAME',
            activation_fn=tf.contrib.keras.layers.LeakyReLU(alpha=0.2),
            normalizer_fn=None if i == 0 else tf.contrib.layers.batch_norm,
            weights_initializer=weights_initializer,
            scope='g_encoder_{}'.format(i),
            reuse=False)

        encoders.append(flow)

        print flow.shape

    # decoder
    for i, k in reversed(list(enumerate(ks))):
        # convolution -> batch norm
        flow = tf.contrib.layers.convolution2d_transpose(
            inputs=flow,
            num_outputs=k,
            kernel_size=4,
            stride=1 if i + 1 == len(ks) else 2,
            padding='SAME',
            normalizer_fn=tf.contrib.layers.batch_norm,
            weights_initializer=weights_initializer,
            scope='g_decoder_{}'.format(i),
            reuse=False)

        # drop out
        if i + 3 >= len(ks):
            flow = tf.contrib.layers.dropout(flow, is_training=is_training)

        # arXiv:1611.07004v1
        # 5.1.1
        # ReLUs in the decoder are not leaky.
        flow = tf.nn.relu(flow)

        # arXiv:1611.07004v1
        # skip connection
        if i + 1 < len(ks):
            flow = tf.concat([flow, encoders[i]], axis=3)

        print flow.shape

    # arXiv:1611.07004v1
    # 5.1.1
    # After the last layer in the decoder, a convolution is applied to map to
    # the number of output channels, followed by a Tanh function.
    flow = tf.contrib.layers.convolution2d_transpose(
        inputs=flow,
        num_outputs=3,
        kernel_size=4,
        stride=2,
        padding='SAME',
        activation_fn=tf.nn.tanh,
        weights_initializer=weights_initializer,
        scope='g_out',
        reuse=False)

    print flow.shape

    return flow


def build_pix2pix(
        source_images, target_images, lambda_value=200.0,
        learning_rate=0.00001, is_training=True):
    """
    build the pix2pix model.
    """
    output_images = build_generator(source_images, is_training)

    if not is_training:
        return {'source_images': source_images, 'output_images': output_images}

    # arXiv:1611.07004v1
    # 2.1
    # Previous approaches to conditional GANs have found it beneficial to mix
    # the GAN objective with a more traditional loss, such as L2 distance.
    # using L1 distance rather than L2 as L1 encourages less blurring.
    l_loss = tf.reduce_mean(tf.abs(target_images - output_images))

    # source and target should be conjugate.
    d_real = build_discriminator(source_images, target_images, False)

    # source and output should not be conjugate, since output is generated.
    d_fake = build_discriminator(source_images, output_images, True)

    # original GAN
    d_temp = d_real * (1.0 - d_fake)
    d_temp = tf.clip_by_value(d_temp, 1e-10, 1.0)
    g_temp = tf.clip_by_value(d_fake, 1e-10, 1.0)

    d_loss = -tf.reduce_mean(tf.log(d_temp))

    # L1 loss for generator
    g_loss = -tf.reduce_mean(tf.log(g_temp)) + lambda_value * l_loss

    #
    global_step = tf.get_variable(
        'global_step',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0, dtype=tf.int64),
        dtype=tf.int64)

    t_vars = tf.trainable_variables()

    d_variables = [v for v in t_vars if v.name.startswith('d_')]
    g_variables = [v for v in t_vars if v.name.startswith('g_')]

    g_trainer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate, beta1=0.5) \
        .minimize(g_loss, var_list=g_variables, global_step=global_step)

    d_trainer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate, beta1=0.5) \
        .minimize(d_loss, var_list=d_variables)

    return {
        'step': global_step,
        'd_loss': d_loss,
        'g_loss': g_loss,
        'd_trainer': d_trainer,
        'g_trainer': g_trainer,

        'source_images': source_images,
        'target_images': target_images,
        'output_images': output_images,
    }
