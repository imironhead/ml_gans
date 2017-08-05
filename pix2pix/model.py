"""
"""
import tensorflow as tf


def build_discriminator(source_images, target_images, reuse=False):
    """
    """
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    flow = tf.concat([source_images, target_images], axis=3)

    # 70 x 70
    ks = [64, 128, 256, 512]

    for i, k in enumerate(ks):
        flow = tf.pad(flow, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')

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

    flow = tf.pad(flow, [[0, 0], [1, 1], [1, 1], [0, 0]], 'CONSTANT')

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
    """
    flow = source_images

    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    ks = [64, 128, 256, 512, 512, 512, 512, 512]

    encoders = []

    print flow.shape

    # encoder
    for i, k in enumerate(ks):
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

        # relu
        flow = tf.nn.relu(flow)

        # skip connection
        if i + 1 < len(ks):
            flow = tf.concat([flow, encoders[i]], axis=3)

        print flow.shape

    # to RGB
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
    """
    output_images = build_generator(source_images, is_training)

    if not is_training:
        return {'source_images': source_images, 'output_images': output_images}

    l_loss = tf.reduce_mean(tf.abs(target_images - output_images))

    d_real = build_discriminator(source_images, target_images, False)
    d_fake = build_discriminator(source_images, output_images, True)

    # original GAN
    d_temp = d_real * (1.0 - d_fake)
    d_temp = tf.clip_by_value(d_temp, 1e-10, 1.0)
    g_temp = tf.clip_by_value(d_fake, 1e-10, 1.0)

    d_loss = -tf.reduce_mean(tf.log(d_temp))
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
