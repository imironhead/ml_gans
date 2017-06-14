"""
"""
import tensorflow as tf

from six.moves import range


tf.app.flags.DEFINE_integer('image-size', 128, '')

# arXiv:1703.10717v2
# 4.1
# we used Nh = Nz = 64 in most of our experiments with this dataset.
tf.app.flags.DEFINE_integer('seed-size', 64, '')
tf.app.flags.DEFINE_integer('embedding-size', 64, '')

# arXiv:1703.10717
# the mysterious n in Figure 1.
tf.app.flags.DEFINE_integer('mysterious-n', 128, '')

# arXiv:1703.10717
# gamma, diversity ratio
tf.app.flags.DEFINE_float('diversity-ratio', 0.7, '')

# arXiv:1703.10717, 3.4
# learning rate of the control variable k.
tf.app.flags.DEFINE_float('k-learning-rate', 0.001, '')

FLAGS = tf.app.flags.FLAGS


def sanity_check():
    """
    """


def build_decoder(flow, scope_prefix, reuse):
    """
    """
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    # arXiv:1703.10717, Figure 1
    # project to [-1, 8x8xn] and reshape to [-1, 8, 8, n] later.
    flow = tf.contrib.layers.fully_connected(
        inputs=flow,
        num_outputs=8 * 8 * FLAGS.mysterious_n,
        activation_fn=tf.nn.elu,
        weights_initializer=weights_initializer,
        scope='{}_project'.format(scope_prefix),
        reuse=reuse)

    # arXiv:1703.10717, Figure 1
    # reshape to [-1, 8, 8, n]
    flow = tf.reshape(flow, [-1, 8, 8, FLAGS.mysterious_n])

    layer_size = 8

    while True:
        for repeat in range(2):
            if layer_size != 8 and repeat == 0:
                # arXiv:1703.10717, Figure 1, upsampling.
                stride = 2
            else:
                # arXiv:1703.10717, Figure 1, repeating.
                stride = 1

            flow = tf.contrib.layers.convolution2d_transpose(
                inputs=flow,
                num_outputs=FLAGS.mysterious_n,
                kernel_size=3,
                stride=stride,
                padding='SAME',
                activation_fn=tf.nn.elu,
                weights_initializer=weights_initializer,
                scope='{}_{}_{}'.format(scope_prefix, layer_size, repeat),
                reuse=reuse)

        if layer_size == FLAGS.image_size:
            break

        layer_size += layer_size

    # arXiv:1703.10717, Figure 1
    # no upsampling and convolve to 3 channels (RGB, [-1.0, +1.0])
    flow = tf.contrib.layers.convolution2d_transpose(
        inputs=flow,
        num_outputs=3,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.tanh,
        weights_initializer=weights_initializer,
        scope='{}_image'.format(scope_prefix),
        reuse=reuse)

    return flow


def build_encoder(flow, scope_prefix, reuse):
    """
    """
    layer_size = FLAGS.image_size
    num_outputs = FLAGS.mysterious_n

    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    while True:
        for repeat in range(2):
            if layer_size != FLAGS.image_size and repeat == 0:
                stride = 2
            else:
                stride = 1

            flow = tf.contrib.layers.convolution2d(
                inputs=flow,
                num_outputs=num_outputs,
                kernel_size=3,
                stride=stride,
                padding='SAME',
                activation_fn=tf.nn.elu,
                weights_initializer=weights_initializer,
                scope='{}_{}_{}'.format(scope_prefix, layer_size, repeat),
                reuse=reuse)

        if layer_size == 8:
            break

        layer_size /= 2
        num_outputs += FLAGS.mysterious_n

    # reshape to embedding
    flow = tf.reshape(flow, [-1, 8 * 8 * num_outputs])

    flow = tf.contrib.layers.fully_connected(
        inputs=flow,
        num_outputs=FLAGS.embedding_size,
        activation_fn=tf.nn.elu,
        weights_initializer=weights_initializer,
        scope='{}_bottleneck'.format(scope_prefix),
        reuse=reuse)

    return flow


def build_discriminator(flow, reuse):
    """
    """
    flow = build_encoder(flow, 'd_encoder', reuse)
    flow = build_decoder(flow, 'd_decoder', reuse)

    return flow


def build_generator(flow):
    """
    """
    return build_decoder(flow, 'g_', False)


def autoencoder_loss(upstream, downstream):
    """
    arXiv:1703.10717v1, L1 norm.

    upstream:
        input images.
    downstream:
        output images. decoded images from autoencoder.
    """
    return tf.reduce_mean(tf.abs(upstream - downstream))


def collect_variables():
    """
    """
    d_variables = []
    g_variables = []

    for variable in tf.trainable_variables():
        if variable.name.startswith('d_'):
            d_variables.append(variable)
        elif variable.name.startswith('g_'):
            g_variables.append(variable)

    return g_variables, d_variables


def build_began(seed, real):
    """
    """
    # global step
    global_step = tf.get_variable(
        'global_step',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0, dtype=tf.int64),
        dtype=tf.int64)

    k = tf.get_variable(
        'k',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0.0, dtype=tf.float32),
        dtype=tf.float32)

    # build the generator to generate images from random seeds.
    fake = build_generator(seed)

    # build the discriminator to judge the real data.
    ae_output_real = build_discriminator(real, False)

    # build the discriminator to judge the fake data.
    # judge both real and fake data with the same network (shared).
    ae_output_fake = build_discriminator(fake, True)

    ae_loss_real = autoencoder_loss(real, ae_output_real)
    ae_loss_fake = autoencoder_loss(fake, ae_output_fake)

    discriminator_loss = ae_loss_real - k * ae_loss_fake

    generator_loss = ae_loss_fake

    # arXiv:1703.10717, 3.4
    # update control variable k
    ae_loss_diff = FLAGS.diversity_ratio * ae_loss_real - ae_loss_fake

    next_k = \
        tf.clip_by_value(k, 0.0, 1.0) + FLAGS.k_learning_rate * ae_loss_diff
    next_k = k.assign(next_k)

    # arXiv:1703.10717, 3.4.1
    # convergence measure, M_global.
    convergence_measure = ae_loss_real + tf.abs(ae_loss_diff)

    #
    learning_rate = tf.get_variable(
        'learning_rate',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0.00005, dtype=tf.float32),
        dtype=tf.float32)

    decay_learning_rate = learning_rate.assign(0.5 * learning_rate)

    g_variables, d_variables = collect_variables()

    generator_trainer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=0.0001)
    generator_trainer = generator_trainer.minimize(
        generator_loss,
        global_step=global_step,
        var_list=g_variables,
        colocate_gradients_with_ops=True)

    discriminator_trainer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=0.0001)
    discriminator_trainer = discriminator_trainer.minimize(
        discriminator_loss,
        var_list=d_variables,
        colocate_gradients_with_ops=True)

    return {
        'global_step': global_step,
        'seed': seed,
        'real': real,
        'fake': fake,
        'next_k': next_k,
        'ae_output_real': ae_output_real,
        'ae_output_fake': ae_output_fake,
        'ae_loss_real': ae_loss_real,
        'ae_loss_fake': ae_loss_fake,
        'generator_loss': generator_loss,
        'generator_trainer': generator_trainer,
        'generator_variables': g_variables,
        'discriminator_loss': discriminator_loss,
        'discriminator_trainer': discriminator_trainer,
        'discriminator_variables': d_variables,
        'convergence_measure': convergence_measure,
        'learning_rate': learning_rate,
        'decay_learning_rate': decay_learning_rate,
    }


def build_embed_network(seed, real):
    """
    """
    fake = build_generator(seed)

    ae_output_fake = build_discriminator(fake, False)

    loss = tf.reduce_mean(tf.abs(fake - real))

    trainer = tf.train.AdamOptimizer(learning_rate=0.0005)
    trainer = trainer.minimize(loss, var_list=[seed])

    g_variables, d_variables = collect_variables()

    return {
        'real': real,
        'fake': fake,
        'loss': loss,
        'seed': seed,
        'ae_output_fake': ae_output_fake,
        'trainer': trainer,
        'generator_variables': g_variables,
        'discriminator_variables': d_variables,
    }


def build_generating_network(seed):
    """
    """
    fake = build_generator(seed)

    ae_output_fake = build_discriminator(fake, False)

    g_variables, d_variables = collect_variables()

    return {
        'fake': fake,
        'seed': seed,
        'ae_output_fake': ae_output_fake,
        'generator_variables': g_variables,
        'discriminator_variables': d_variables,
    }
