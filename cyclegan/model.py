"""
"""
import tensorflow as tf

from six.moves import range


def build_c_layer(flow, scope, reuse, num_outputs=3, activation_fn=tf.nn.relu):
    """
    arXiv:1703.10593v1
    c7s1-32
    """
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    flow = tf.pad(
        tensor=flow,
        paddings=[[0, 0], [3, 3], [3, 3], [0, 0]],
        mode='reflect')

    flow = tf.contrib.layers.convolution2d(
        inputs=flow,
        num_outputs=num_outputs,
        kernel_size=7,
        stride=1,
        padding='VALID',
        activation_fn=activation_fn,
        normalizer_fn=instance_norm,
        weights_initializer=weights_initializer,
        scope=scope,
        reuse=reuse)

    return flow


def build_d_layer(flow, scope, reuse, num_outputs=3):
    """
    arXiv:1703.10593v1
    d64
    """
    # NOTE: kernel size on paper is 3x3

    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    flow = tf.pad(
        tensor=flow,
        paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
        mode='reflect')

    flow = tf.contrib.layers.convolution2d(
        inputs=flow,
        num_outputs=num_outputs,
        kernel_size=4,
        stride=2,
        padding='VALID',
        activation_fn=tf.nn.relu,
        normalizer_fn=instance_norm,
        weights_initializer=weights_initializer,
        scope=scope,
        reuse=reuse)

    return flow


def build_r_layer(flow, scope, reuse, num_outputs=128):
    """
    arXiv:1703.10593v1
    R128
    """
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    # FIXME: padding

    flow_input = flow

    flow = tf.pad(
        tensor=flow,
        paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
        mode='reflect')

    flow = tf.contrib.layers.convolution2d(
        inputs=flow,
        num_outputs=num_outputs,
        kernel_size=3,
        stride=1,
        padding='VALID',
        activation_fn=tf.nn.relu,
        normalizer_fn=instance_norm,
        weights_initializer=weights_initializer,
        scope='{}-R-{}-1'.format(scope, num_outputs),
        reuse=reuse)

    flow = tf.pad(
        tensor=flow,
        paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
        mode='reflect')

    flow = tf.contrib.layers.convolution2d(
        inputs=flow,
        num_outputs=num_outputs,
        kernel_size=3,
        stride=1,
        padding='VALID',
        weights_initializer=weights_initializer,
        scope='{}-R-{}-2'.format(scope, num_outputs),
        reuse=reuse)

    flow = flow + flow_input

    return flow


def build_u_layer(flow, scope, reuse, num_outputs=3):
    """
    arXiv:1703.10593v1
    u64
    """
    # NOTE: kernel size on paper is 3x3

    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    # flow = tf.pad(
    #     tensor=flow,
    #     paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
    #     mode='reflect')

    flow = tf.contrib.layers.convolution2d_transpose(
        inputs=flow,
        num_outputs=num_outputs,
        kernel_size=4,
        stride=2,
        # padding='VALID',
        padding='SAME',
        activation_fn=tf.nn.relu,
        normalizer_fn=instance_norm,
        weights_initializer=weights_initializer,
        scope=scope,
        reuse=reuse)

    return flow


def instance_norm(flow):
    """
    arXiv:1607.08022v2
    """
    mean, variance = tf.nn.moments(flow, axes=[1, 2], keep_dims=True)

    return (flow - mean) / tf.sqrt(variance + 1e-3)


def build_discriminator(images, prefix, reuse=False):
    """
    arXiv:1703.10593v1
    build 70x70 PatchGAN discriminator
    """
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    flow = images

    # arXiv:1703.10593v1
    # 4. Implementation: Network Architecture
    # 70 x 70 discriminator
    ks = [64, 128, 256, 512]

    for i, k in enumerate(ks):
        # arXiv:1611.07004v1
        # we achieve this variation in patch size by adjusting the depth of the
        # GAN discriminator.

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
            normalizer_fn=None if i == 0 else instance_norm,
            weights_initializer=weights_initializer,
            scope='{}_{}_{}'.format(prefix, i, k),
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
        scope='{}_out'.format(prefix),
        reuse=reuse)

    return flow


def print_tensor(tag, flow):
    """
    """
    print(tag, flow.shape)


def build_generator(source_images, prefix, reuse=False):
    """
    arXiv:1611.07004v1
    build encoder-decoder generator
    """
    flow = source_images

    # arXiv:1703.10593v1
    # 9 blocks for 256x256
    # c7s1-32,
    # d64, d128,
    # R128, R128, R128, R128, R128, R128, R128, R128, R128,
    # u64, u32
    # c7s1-3
    print_tensor('input', flow)

    flow = build_c_layer(flow, prefix + '-c7s1-32', reuse, num_outputs=32)

    print_tensor(prefix + '-c7s1-32', flow)

    flow = build_d_layer(flow, prefix + '-d64', reuse, num_outputs=64)

    print_tensor(prefix + '-d64', flow)

    flow = build_d_layer(flow, prefix + '-d128', reuse, num_outputs=128)

    print_tensor(prefix + '-d128', flow)

    for i in range(9):
        flow = build_r_layer(
            flow, prefix + '-r128-{}'.format(i), reuse, num_outputs=128)

        print_tensor(prefix + '-r128-{}'.format(i), flow)

    flow = build_u_layer(flow, prefix + '-u64', reuse, num_outputs=64)

    print_tensor(prefix + '-u64', flow)

    flow = build_u_layer(flow, prefix + '-u32', reuse, num_outputs=32)

    print_tensor(prefix + '-u32', flow)

    flow = build_c_layer(
        flow, prefix + '-c7s1-3', reuse, num_outputs=3,
        activation_fn=tf.nn.tanh)

    print_tensor(prefix + '-c7s1-3', flow)

    return flow


def build_cycle_gan(x_images, y_images, gx_history, fy_history, is_training):
    """
    """
    gx_fake = build_generator(x_images, 'gx_', reuse=False)
    fy_fake = build_generator(y_images, 'fy_', reuse=False)
    gf_cycl = build_generator(fy_fake, 'gx_', reuse=True)
    fg_cycl = build_generator(gx_fake, 'fy_', reuse=True)

    dx_real = build_discriminator(x_images, 'dx_', reuse=False)
    dx_dirt = build_discriminator(fy_history, 'dx_', reuse=True)
    dx_fake = build_discriminator(fy_fake, 'dx_', reuse=True)

    dy_real = build_discriminator(y_images, 'dy_', reuse=False)
    dy_dirt = build_discriminator(gx_history, 'dy_', reuse=True)
    dy_fake = build_discriminator(gx_fake, 'dy_', reuse=True)

    # loss_dx = tf.nn.l2_loss(dx_real - 1.0) + tf.nn.l2_loss(dx_dirt)
    loss_dx = tf.nn.l2_loss(dx_real - 1.0) + tf.nn.l2_loss(dx_fake)
    loss_dx = tf.reduce_mean(loss_dx)

    loss_fy = tf.nn.l2_loss(dx_fake - 1.0) + 10.0 * tf.abs(gf_cycl - y_images)
    loss_fy = tf.reduce_mean(loss_fy)

    # loss_dy = tf.nn.l2_loss(dy_real - 1.0) + tf.nn.l2_loss(dy_dirt)
    loss_dy = tf.nn.l2_loss(dy_real - 1.0) + tf.nn.l2_loss(dy_fake)
    loss_dy = tf.reduce_mean(loss_dy)

    loss_gx = tf.nn.l2_loss(dy_fake - 1.0) + 10.0 * tf.abs(fg_cycl - x_images)
    loss_gx = tf.reduce_mean(loss_gx)

    #
    step = tf.get_variable(
        'global_step',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0, dtype=tf.int64),
        dtype=tf.int64)

    # collect trainable variables
    t_vars = tf.trainable_variables()

    dx_vars = [v for v in t_vars if v.name.startswith('dx_')]
    dy_vars = [v for v in t_vars if v.name.startswith('dy_')]
    fy_vars = [v for v in t_vars if v.name.startswith('fy_')]
    gx_vars = [v for v in t_vars if v.name.startswith('gx_')]

    fy_trainer = tf.train \
        .AdamOptimizer(learning_rate=0.0002, beta1=0.5) \
        .minimize(loss_fy, var_list=fy_vars, global_step=step)

    gx_trainer = tf.train \
        .AdamOptimizer(learning_rate=0.0002, beta1=0.5) \
        .minimize(loss_gx, var_list=gx_vars)

    dx_trainer = tf.train \
        .AdamOptimizer(learning_rate=0.0002, beta1=0.5) \
        .minimize(loss_dx, var_list=dx_vars)

    dy_trainer = tf.train \
        .AdamOptimizer(learning_rate=0.0002, beta1=0.5) \
        .minimize(loss_dy, var_list=dy_vars)

    return {
        'step': step,

        'x_images': x_images,
        'y_images': y_images,
        'gx_images': gx_fake,
        'fy_images': fy_fake,

        'dx_trainer': dx_trainer,
        'dy_trainer': dy_trainer,
        'fy_trainer': fy_trainer,
        'gx_trainer': gx_trainer,
    }
