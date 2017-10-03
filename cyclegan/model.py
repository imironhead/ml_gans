"""
"""
import tensorflow as tf

from six.moves import range


def leaky_relu(flow, slope):
    """
    """
    return tf.maximum(flow, flow * slope)


def build_c_layer(flow, name, num_outputs=3, activation_fn=tf.nn.relu):
    """
    arXiv:1703.10593v1
    c7s1-32
    """
    with tf.variable_scope(name):
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
            activation_fn=None,
            weights_initializer=weights_initializer,
            biases_initializer=None)

        # if activation_fn != tf.nn.tanh:
        if num_outputs != 3:
            flow = instance_norm(flow, 'norm')

        flow = activation_fn(flow)

    return flow


def build_d_layer(flow, name, num_outputs=3):
    """
    arXiv:1703.10593v1
    d64
    """
    with tf.variable_scope(name):
        weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

        flow = tf.contrib.layers.convolution2d(
            inputs=flow,
            num_outputs=num_outputs,
            kernel_size=3,
            stride=2,
            padding='SAME',
            activation_fn=None,
            weights_initializer=weights_initializer,
            biases_initializer=None)

        flow = instance_norm(flow, 'norm')

        flow = tf.nn.relu(flow)

    return flow


def build_r_layer(flow, name, num_outputs=128):
    """
    arXiv:1703.10593v1
    R128
    """
    with tf.variable_scope(name):
        weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

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
            activation_fn=None,
            weights_initializer=weights_initializer,
            biases_initializer=None,
            scope='conv1')

        flow = instance_norm(flow, 'norm1')

        flow = tf.nn.relu(flow)

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
            activation_fn=None,
            weights_initializer=weights_initializer,
            biases_initializer=None,
            scope='conv2')

        flow = instance_norm(flow, 'norm2')

        flow = flow + flow_input

    return flow


def build_u_layer(flow, name, num_outputs=3):
    """
    arXiv:1703.10593v1
    u64
    """
    with tf.variable_scope(name):
        weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

        flow = tf.contrib.layers.convolution2d_transpose(
            inputs=flow,
            num_outputs=num_outputs,
            kernel_size=3,
            stride=2,
            padding='SAME',
            activation_fn=None,
            weights_initializer=weights_initializer,
            biases_initializer=None,
            scope='convt')

        flow = instance_norm(flow, 'norm')

        flow = tf.nn.relu(flow)

    return flow


def instance_norm(flow, name):
    """
    arXiv:1607.08022v2
    """
    with tf.variable_scope(name):
        depth = flow.get_shape()[3]
        scale = tf.get_variable(
            'scale',
            [depth],
            initializer=tf.random_normal_initializer(1.0, 0.02))
        shift = tf.get_variable(
            'shift',
            [depth],
            initializer=tf.constant_initializer(0.0))

        mean, variance = tf.nn.moments(flow, axes=[1, 2], keep_dims=True)

        flow = (flow - mean) / tf.sqrt(variance + 1e-5)

        return scale * flow + shift


def build_discriminator(flow, name, reuse=None):
    """
    arXiv:1703.10593v1
    build 70x70 PatchGAN discriminator
    """
    with tf.variable_scope(name, reuse=reuse):
        weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

        # arXiv:1703.10593v1
        # 4. Implementation: Network Architecture
        # 70 x 70 discriminator
        ks = [64, 128, 256, 512]

        for i, k in enumerate(ks):
            # arXiv:1611.07004v1
            # we achieve this variation in patch size by adjusting the depth of
            # the GAN discriminator.
            flow = tf.contrib.layers.convolution2d(
                inputs=flow,
                num_outputs=k,
                kernel_size=4,
                stride=2 if i != 3 else 1,
                padding='SAME',
                activation_fn=None,
                weights_initializer=weights_initializer,
                biases_initializer=None,
                scope='conv_{}_{}'.format(i, k))

            # arXiv:1611.07004v1
            # 5.1.2
            # As an exception to the above notation, BatchNorm is not applied
            # to the first C64 layer. All ReLU are leaky, with slope 0.2.
            if i > 0:
                flow = instance_norm(flow, 'norm_{}'.format(i))

            flow = leaky_relu(flow, 0.2)

        # arXiv:1611.07004v1
        # 5.1.2
        # after the last layer, a convolution is applied to map to a 1
        # dimensional output, followed by a Sigmoid function.
        flow = tf.contrib.layers.convolution2d(
            inputs=flow,
            num_outputs=1,
            kernel_size=4,
            stride=1,
            padding='SAME',
            activation_fn=None,
            weights_initializer=weights_initializer,
            biases_initializer=None,
            scope='out')

    return flow


def print_tensor(flow):
    """
    """
    print('{} - {}'.format(flow.name, flow.shape))


def build_generator(flow, name, reuse=None):
    """
    arXiv:1611.07004v1
    build encoder-decoder generator
    """
    # arXiv:1703.10593v1
    # 9 blocks for 256x256
    # c7s1-32,
    # d64, d128,
    # R128, R128, R128, R128, R128, R128, R128, R128, R128,
    # u64, u32
    # c7s1-3
    with tf.variable_scope(name, reuse=reuse):
        print_tensor(flow)

        flow = build_c_layer(flow, 'c7s1-32', num_outputs=32)

        print_tensor(flow)

        flow = build_d_layer(flow, 'd64', num_outputs=64)

        print_tensor(flow)

        flow = build_d_layer(flow, 'd128', num_outputs=128)

        print_tensor(flow)

        for i in range(9):
            flow = build_r_layer(flow, 'r128-{}'.format(i), num_outputs=128)

            print_tensor(flow)

        flow = build_u_layer(flow, 'u64', num_outputs=64)

        print_tensor(flow)

        flow = build_u_layer(flow, 'u32', num_outputs=32)

        print_tensor(flow)

        flow = build_c_layer(
            flow, 'c7s1-3', num_outputs=3, activation_fn=tf.nn.tanh)

    print_tensor(flow)

    return flow


def build_cycle_gan(xx_real, yy_real, is_training):
    """
    """
    gx_pool = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32)
    fy_pool = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32)

    gx_fake = build_generator(xx_real, 'gx_')
    fy_fake = build_generator(yy_real, 'fy_')
    gf_cycl = build_generator(fy_fake, 'gx_', reuse=True)
    fg_cycl = build_generator(gx_fake, 'fy_', reuse=True)

    dx_real = build_discriminator(xx_real, 'dx_')
    dx_pool = build_discriminator(fy_pool, 'dx_', reuse=True)
    dx_fake = build_discriminator(fy_fake, 'dx_', reuse=True)

    dy_real = build_discriminator(yy_real, 'dy_')
    dy_pool = build_discriminator(gx_pool, 'dy_', reuse=True)
    dy_fake = build_discriminator(gx_fake, 'dy_', reuse=True)

    loss_dx = \
        tf.reduce_mean((dx_real - 1.0) ** 2.0) + \
        tf.reduce_mean(dx_pool ** 2.0)

    loss_dy = \
        tf.reduce_mean((dy_real - 1.0) ** 2.0) + \
        tf.reduce_mean(dy_pool ** 2.0)

    loss_fy = \
        tf.reduce_mean((dx_fake - 1.0) ** 2.0) + \
        tf.reduce_mean(tf.abs(gf_cycl - yy_real)) * 10.0

    loss_gx = \
        tf.reduce_mean((dy_fake - 1.0) ** 2.0) + \
        tf.reduce_mean(tf.abs(fg_cycl - xx_real)) * 10.0

    loss_d = loss_dx + loss_dy
    loss_g = loss_gx + loss_fy

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

    d_vars = dx_vars + dy_vars
    g_vars = gx_vars + fy_vars

    learning_rate = tf.get_variable(
        'learning_rate',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0.0002, dtype=tf.float32),
        dtype=tf.float32)

    g_trainer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate, beta1=0.5) \
        .minimize(loss_g, var_list=g_vars, global_step=step)

    d_trainer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate, beta1=0.5) \
        .minimize(loss_d, var_list=d_vars)

    return {
        'step': step,

        'xx_real': xx_real,
        'yy_real': yy_real,
        'gx_fake': gx_fake,
        'fy_fake': fy_fake,
        'gx_pool': gx_pool,
        'fy_pool': fy_pool,

        'g_trainer': g_trainer,
        'd_trainer': d_trainer,

        'loss_d': loss_d,
        'loss_g': loss_g,
        'loss_dx': loss_dx,
        'loss_dy': loss_dy,
        'loss_gx': loss_gx,
        'loss_fy': loss_fy,

        'learning_rate': learning_rate,
    }
