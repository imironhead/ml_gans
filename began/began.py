"""
"""
import glob
import numpy as np
import os
import tensorflow as tf

from six.moves import range


tf.app.flags.DEFINE_string('logs-dir-path', './began/logs/', '')
tf.app.flags.DEFINE_string('checkpoints-dir-path', './began/checkpoints/', '')

# arXiv:1703.10717
# we typically used a batch size of n = 16.
tf.app.flags.DEFINE_integer('batch-size', 16, '')

tf.app.flags.DEFINE_integer('image-size', 64, '')

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

tf.app.flags.DEFINE_integer('summary-row-size', 4, '')
tf.app.flags.DEFINE_integer('summary-col-size', 4, '')

FLAGS = tf.app.flags.FLAGS


def sanity_check():
    """
    """
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
    layer_size = 8

    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

    flow = tf.contrib.layers.fully_connected(
        inputs=flow,
        num_outputs=layer_size * layer_size * FLAGS.mysterious_n,
        activation_fn=tf.nn.elu,
        weights_initializer=weights_initializer,
        scope='{}_project'.format(scope_prefix),
        reuse=reuse)

    # reshape to images
    flow = tf.reshape(flow, [-1, layer_size, layer_size, FLAGS.mysterious_n])

    while True:
        for repeat in range(2):
            if layer_size != 8 and repeat == 0:
                stride = 2
            else:
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
    l = FLAGS.image_size * FLAGS.image_size * 3
    k = tf.reshape(upstream - downstream, [-1, l])
    k = tf.norm(k, 1, axis=1)

    return tf.reduce_mean(k)


def build_dataset_reader():
    """
    """
    paths_png = glob.glob(
        '/home/ironhead/datasets/celeba/img_align_celeba_png/*.png')

    file_name_queue = tf.train.string_input_producer(paths_png)

    reader = tf.WholeFileReader()

    reader_key, reader_val = reader.read(file_name_queue)

    image = tf.image.decode_png(reader_val, channels=3, dtype=tf.uint8)

    image = tf.image.crop_to_bounding_box(image, 50, 25, 128, 128)

    image = tf.image.resize_images(image, [64, 64])

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

    # the input batch placeholder (real data) for the discriminator.
    real = build_dataset_reader()

    # the input batch placeholder for the generator.
    seed = tf.random_uniform(
        shape=[FLAGS.batch_size, FLAGS.seed_size], minval=-1.0, maxval=1.0)

    # build the generator to generate images from random seeds.
    fake = build_generator(seed)

    # build the discriminator to judge the real data.
    discriminate_real = build_discriminator(real, False)

    # build the discriminator to judge the fake data.
    # judge both real and fake data with the same network (shared).
    discriminate_fake = build_discriminator(fake, True)

    lx = autoencoder_loss(real, discriminate_real)
    lg = autoencoder_loss(fake, discriminate_fake)

    discriminator_loss = lx - k * lg

    generator_loss = lg

    q = k + 0.001 * (FLAGS.diversity_ratio * lx - lg)

    q = tf.clip_by_value(q, 0.0, 1.0)

    update_k = k.assign(q)

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
        'generator_fake': fake,
        'generator_loss': generator_loss,
        'generator_trainer': generator_trainer,
        'discriminator_loss': discriminator_loss,
        'discriminator_trainer': discriminator_trainer,
        'update_k': update_k,
        'loss_real': lx,
        'loss_fake': lg,
        'discriminate_real': discriminate_real,
    }


def build_summaries(gan):
    """
    """
    generator_loss_summary = tf.summary.scalar(
        'generator loss', gan['generator_loss'])

    discriminator_loss_summary = tf.summary.scalar(
        'discriminator loss', gan['discriminator_loss'])

    fake_grid = tf.reshape(gan['generator_fake'], [1, 16 * 64, 64, 3])
    fake_grid = tf.split(fake_grid, 4, axis=1)
    fake_grid = tf.concat(fake_grid, axis=2)
    fake_grid = tf.saturate_cast(fake_grid * 127.5 + 127.5, tf.uint8)

    generator_fake_summary = tf.summary.image(
        'generated image', fake_grid, max_outputs=4)

    temp_grid = tf.reshape(gan['discriminate_real'], [1, 16 * 64, 64, 3])
    temp_grid = tf.split(temp_grid, 4, axis=1)
    temp_grid = tf.concat(temp_grid, axis=2)
    temp_grid = tf.saturate_cast(temp_grid * 127.5 + 127.5, tf.uint8)

    discriminator_temp_summary = tf.summary.image(
        'autoencoder image', temp_grid, max_outputs=4)

    return {
        'generator_fake_summary': generator_fake_summary,
        'generator_loss_summary': generator_loss_summary,
        'discriminator_loss_summary': discriminator_loss_summary,
        'discriminator_temp_summary': discriminator_temp_summary,
    }


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
            # for _ in range(5):
                # real_sources = next_real_batch(reader)
                # fake_sources = next_fake_batch()

            fetches = [
                gan_graph['discriminator_loss'],
                gan_graph['discriminator_trainer'],
                summaries['discriminator_loss_summary'],
                gan_graph['update_k'],
                gan_graph['loss_real'],
            ]

            feeds = {
                # gan_graph['seed']: fake_sources,
                # gan_graph['real']: real_sources,
            }

            if global_step % 500 == 0:
                fetches.append(summaries['discriminator_temp_summary'])

            returns = session.run(fetches, feed_dict=feeds)

            d_loss_summary = returns[2]

            loss_real = returns[4]

            reporter.add_summary(d_loss_summary, global_step)

            if global_step % 500 == 0:
                reporter.add_summary(returns[5], global_step)

            #
            # fake_sources = next_fake_batch()

            log_fakes = (global_step % 500 == 0)

            fetches = [
                gan_graph['global_step'],
                gan_graph['generator_loss'],
                gan_graph['generator_trainer'],
                summaries['generator_loss_summary'],
                gan_graph['loss_fake'],
            ]

            feeds = {}

            if log_fakes:
                fetches.append(summaries['generator_fake_summary'])

            returns = session.run(fetches, feed_dict=feeds)

            global_step = returns[0]
            g_loss_summary = returns[3]

            reporter.add_summary(g_loss_summary, global_step)

            if log_fakes:
                reporter.add_summary(returns[5], global_step)

            if global_step % 100 == 0:
                loss_fake = returns[4]

                mg = loss_real + np.abs(FLAGS.diversity_ratio * loss_real - loss_fake)

                print('[{}]: {}, {}'.format(global_step, returns[1], mg))

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
