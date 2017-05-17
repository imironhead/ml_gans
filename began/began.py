"""
"""
import glob
import os
import tensorflow as tf

from six.moves import range


tf.app.flags.DEFINE_string('portraits-dir-path', '', '')
tf.app.flags.DEFINE_string('logs-dir-path', './began/logs/', '')
tf.app.flags.DEFINE_string('checkpoints-dir-path', './began/checkpoints/', '')

# arXiv:1703.10717
# we typically used a batch size of n = 16.
tf.app.flags.DEFINE_integer('batch-size', 16, '')

tf.app.flags.DEFINE_integer('image-size', 64, '')

tf.app.flags.DEFINE_boolean('need-crop-image', False, '')
tf.app.flags.DEFINE_integer('image-offset-x', 25, '')
tf.app.flags.DEFINE_integer('image-offset-y', 50, '')

# arXiv:1703.10717v2
# 4.1
# we used Nh = Nz = 64 in most of our experiments with this dataset.
tf.app.flags.DEFINE_integer('seed-size', 64, '')
tf.app.flags.DEFINE_integer('embedding-size', 64, '')

# arXiv:1703.10717
# the mysterious n in Figure 1.
tf.app.flags.DEFINE_integer('mysterious-n', 64, '')

# arXiv:1703.10717
# gamma, diversity ratio
tf.app.flags.DEFINE_float('diversity-ratio', 0.7, '')

# arXiv:1703.10717, 3.4
# learning rate of the control variable k.
tf.app.flags.DEFINE_float('k-learning-rate', 0.001, '')

tf.app.flags.DEFINE_integer('summary-row-size', 4, '')
tf.app.flags.DEFINE_integer('summary-col-size', 4, '')

FLAGS = tf.app.flags.FLAGS


def sanity_check():
    """
    """
    if not os.path.isdir(FLAGS.portraits_dir_path):
        raise Exception('invalid portraits directory')

    run_name = '{}_{}_{}_{}'.format(
        FLAGS.seed_size, FLAGS.embedding_size, FLAGS.image_size,
        FLAGS.mysterious_n)

    FLAGS.logs_dir_path = \
        os.path.join(FLAGS.logs_dir_path, run_name)
    FLAGS.checkpoints_dir_path = \
        os.path.join(FLAGS.checkpoints_dir_path, run_name)

    if not os.path.isdir(FLAGS.logs_dir_path):
        os.makedirs(FLAGS.logs_dir_path)

    if not os.path.isdir(FLAGS.checkpoints_dir_path):
        os.makedirs(FLAGS.checkpoints_dir_path)


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


def build_dataset_reader():
    """
    """
    paths_png_wildcards = os.path.join(FLAGS.portraits_dir_path, '*.png')

    paths_png = glob.glob(paths_png_wildcards)

    file_name_queue = tf.train.string_input_producer(paths_png)

    reader = tf.WholeFileReader()

    reader_key, reader_val = reader.read(file_name_queue)

    image = tf.image.decode_png(reader_val, channels=3, dtype=tf.uint8)

    # assume the size of input images are either 128x128x3 or 64x64x3.

    if FLAGS.need_crop_image:
        image = tf.image.crop_to_bounding_box(
            image,
            FLAGS.image_offset_y,
            FLAGS.image_offset_x,
            FLAGS.image_size,
            FLAGS.image_size)

    image = tf.image.resize_images(image, [FLAGS.image_size, FLAGS.image_size])

    image = tf.cast(image, dtype=tf.float32) / 127.5 - 1.0

    return tf.train.batch(
        tensors=[image],
        batch_size=FLAGS.batch_size,
        capacity=FLAGS.batch_size)


def build_began():
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

    # the input batch (real data) for the discriminator.
    real = build_dataset_reader()

    # the input batch (uniform z) for the generator.
    seed = tf.random_uniform(
        shape=[FLAGS.batch_size, FLAGS.seed_size], minval=-1.0, maxval=1.0)

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

    next_k = k + FLAGS.k_learning_rate * ae_loss_diff
    next_k = tf.clip_by_value(next_k, 0.0, 1.0)
    next_k = k.assign(next_k)

    # arXiv:1703.10717, 3.4.1
    # convergence measure, M_global.
    convergence_measure = ae_loss_real + tf.abs(ae_loss_diff)

    #
    d_variables = []
    g_variables = []

    for variable in tf.trainable_variables():
        if variable.name.startswith('d_'):
            d_variables.append(variable)
        elif variable.name.startswith('g_'):
            g_variables.append(variable)

    generator_trainer = tf.train.AdamOptimizer(
        learning_rate=0.00005, beta1=0.0001)
    generator_trainer = generator_trainer.minimize(
        generator_loss,
        global_step=global_step,
        var_list=g_variables,
        colocate_gradients_with_ops=True)

    discriminator_trainer = tf.train.AdamOptimizer(
        learning_rate=0.00005, beta1=0.0001)
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
        'discriminator_loss': discriminator_loss,
        'discriminator_trainer': discriminator_trainer,
        'convergence_measure': convergence_measure,
    }


def reshape_batch_images(batch_images):
    """
    """
    batch_size = FLAGS.batch_size
    image_size = FLAGS.image_size

    # build summary for generated fake images.
    grid = \
        tf.reshape(batch_images, [1, batch_size * image_size, image_size, 3])
    grid = tf.split(grid, FLAGS.summary_row_size, axis=1)
    grid = tf.concat(grid, axis=2)
    grid = tf.saturate_cast(grid * 127.5 + 127.5, tf.uint8)

    return grid


def build_summaries(gan):
    """
    """
    summaries = {}

    # build scalar summaries
    scalar_table = [
        ('summary_convergence_measure', 'convergence_measure',
         'convergence measure'),
        ('summary_generator_loss', 'generator_loss', 'generator loss'),
        ('summary_discriminator_loss', 'discriminator_loss',
         'discriminator loss'),
    ]

    for scalar in scalar_table:
        summaries[scalar[0]] = tf.summary.scalar(scalar[2], gan[scalar[1]])

    # build image summaries
    image_table = [
        ('summary_real', 'real', 'real image'),
        ('summary_fake', 'fake', 'generated image'),
        ('summary_ae_real', 'ae_output_real', 'autoencoder real'),
        ('summary_ae_fake', 'ae_output_fake', 'autoencoder fake')
    ]

    for table in image_table:
        grid = reshape_batch_images(gan[table[1]])

        summaries[table[0]] = tf.summary.image(table[2], grid, max_outputs=4)

    return summaries


def train():
    """
    """
    # tensorflow
    checkpoint_source_path = tf.train.latest_checkpoint(
        FLAGS.checkpoints_dir_path)
    checkpoint_target_path = os.path.join(
        FLAGS.checkpoints_dir_path, 'model.ckpt')

    gan_graph = build_began()
    summaries = build_summaries(gan_graph)

    reporter = tf.summary.FileWriter(FLAGS.logs_dir_path)

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if checkpoint_source_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, checkpoint_source_path)

        # give up overlapped old data
        global_step = session.run(gan_graph['global_step'])

        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START),
            global_step=global_step)

        while True:
            # discriminator
            fetches = [
                gan_graph['next_k'],
                gan_graph['discriminator_trainer'],
                summaries['summary_convergence_measure'],
                summaries['summary_discriminator_loss'],
            ]

            if global_step % 500 == 0:
                fetches.append(summaries['summary_fake'])
                fetches.append(summaries['summary_real'])
                fetches.append(summaries['summary_ae_fake'])
                fetches.append(summaries['summary_ae_real'])

            returns = session.run(fetches)

            for summary in returns[2:]:
                reporter.add_summary(summary, global_step)

            # generator
            fetches = [
                gan_graph['generator_trainer'],
                gan_graph['global_step'],
                summaries['summary_generator_loss'],
            ]

            returns = session.run(fetches)

            global_step = returns[1]

            reporter.add_summary(returns[2], global_step)

            if global_step % 100 == 0:
                print('step {}'.format(global_step))

            if global_step % 5000 == 0:
                tf.train.Saver().save(
                    session,
                    checkpoint_target_path,
                    global_step=gan_graph['global_step'])

        coord.request_stop()
        coord.join(threads)


def main(_):
    """
    """
    sanity_check()
    train()


if __name__ == '__main__':
    """
    """
    tf.app.run()
