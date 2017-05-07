"""
"""
import numpy as np
import ore
import os
import tensorflow as tf
import time

from dcgan_lsun import discriminator, generator
from six.moves import range


tf.app.flags.DEFINE_string(
    'logs-dir-path', './xwgan/logs/lsun/', '')
tf.app.flags.DEFINE_string(
    'outputs-dir-path', './xwgan/outputs/', '')
tf.app.flags.DEFINE_string(
    'checkpoints-dir-path', './xwgan/checkpoints/lsun/', '')
tf.app.flags.DEFINE_integer('batch-size', 64, '')
tf.app.flags.DEFINE_integer('seed-size', 128, '')
tf.app.flags.DEFINE_integer('summary-row-size', 8, '')
tf.app.flags.DEFINE_integer('summary-col-size', 8, '')
tf.app.flags.DEFINE_boolean('generate', False, '')

FLAGS = tf.app.flags.FLAGS


def sanity_check():
    """
    """
    if not tf.gfile.Exists(FLAGS.logs_dir_path):
        raise Exception(
            'bad logs dir path: {}'.format(FLAGS.logs_dir_path))

    if not tf.gfile.Exists(FLAGS.outputs_dir_path):
        raise Exception(
            'bad outputs dir path: {}'.format(FLAGS.outputs_dir_path))

    if not tf.gfile.Exists(FLAGS.checkpoints_dir_path):
        raise Exception(
            'bad checkpoints dir path: {}'.format(FLAGS.checkpoints_dir_path))

    if FLAGS.batch_size < 1:
        raise Exception('bad batch size: {}'.format(FLAGS.batch_size))

    if FLAGS.seed_size < 1:
        raise Exception('bad seed size: {}'.format(FLAGS.seed_size))

    if FLAGS.summary_row_size < 1:
        raise Exception(
            'bad summary row size: {}'.format(FLAGS.summary_row_size))

    if FLAGS.summary_col_size < 1:
        raise Exception(
            'bad summary col size: {}'.format(FLAGS.summary_col_size))

    if FLAGS.summary_col_size * FLAGS.summary_row_size != FLAGS.batch_size:
        message = '{} x {} != {}'.format(
            FLAGS.summary_col_size, FLAGS.summary_row_size, FLAGS.batch_size)

        raise Exception('bad summary size: {}'.format(message))


def build_xwgan():
    """
    """
    # global step
    global_step = tf.get_variable(
        "gstep",
        [],
        trainable=False,
        initializer=tf.constant_initializer(0.0))

    # the input batch placeholder for the generator.
    seed = tf.placeholder(shape=[None, FLAGS.seed_size], dtype=tf.float32)

    # the input batch placeholder (real data) for the discriminator.
    real = tf.placeholder(shape=[None, 64, 64, 3], dtype=tf.float32)

    # build the generator to generate images from random seeds.
    fake = generator(seed)

    # build the discriminator to judge the real data.
    discriminate_real = discriminator(real, False)

    # build the discriminator to judge the fake data.
    # judge both real and fake data with the same network (shared).
    discriminate_fake = discriminator(fake, True)

    # gradient penalty
    alpha = tf.random_uniform([tf.shape(seed)[0], 1, 1, 1])
    inter = fake + alpha * (real - fake)

    discriminate_inte = discriminator(inter, True)

    gradients = tf.gradients(discriminate_inte, inter)[0]

    gradients_norm = tf.norm(gradients, 2, axis=1)

    gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2.0)

    # loss
    discriminator_loss = tf.reduce_mean(
        discriminate_fake - discriminate_real) + 1.0 * gradient_penalty

    generator_loss = -tf.reduce_mean(discriminate_fake)

    #
    d_variables = []
    g_variables = []

    for variable in tf.trainable_variables():
        if variable.name.startswith('d_'):
            d_variables.append(variable)
        elif variable.name.startswith('g_'):
            g_variables.append(variable)

    generator_trainer = tf.train.AdamOptimizer(
        learning_rate=0.0001, beta1=0.5, beta2=0.9)
    generator_trainer = generator_trainer.minimize(
        generator_loss,
        global_step=global_step,
        var_list=g_variables,
        colocate_gradients_with_ops=True)

    discriminator_trainer = tf.train.AdamOptimizer(
        learning_rate=0.0001, beta1=0.5, beta2=0.9)
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
    }


def build_summaries(gan_graph):
    """
    """
    generator_loss_summary = tf.summary.scalar(
        'generator loss', gan_graph['generator_loss'])

    discriminator_loss_summary = tf.summary.scalar(
        'discriminator loss', gan_graph['discriminator_loss'])

    fake_grid = tf.reshape(gan_graph['generator_fake'], [1, 64 * 64, 64, 3])
    fake_grid = tf.split(fake_grid, 8, axis=1)
    fake_grid = tf.concat(fake_grid, axis=2)
    fake_grid = tf.saturate_cast(fake_grid * 127.5 + 127.5, tf.uint8)

    generator_fake_summary = tf.summary.image(
        'generated image', fake_grid, max_outputs=1)

    return {
        'generated_png': tf.image.encode_png(fake_grid[0]),
        'generator_fake_summary': generator_fake_summary,
        'generator_loss_summary': generator_loss_summary,
        'discriminator_loss_summary': discriminator_loss_summary,
    }


def next_real_batch(reader):
    """
    Get next batch from LSUN. A bunch of RGB images. The images are resized to
    [batch_size, 64, 64, 3].
    """
    raw_batch, _, _ = reader.next_batch(FLAGS.batch_size)

    # crop to 64 x 64 x 3 and copy to numpy array
    discriminator_batch = np.zeros((FLAGS.batch_size, 64, 64, 3))

    for i in range(FLAGS.batch_size):
        image = raw_batch[i]

        w, h = image.shape[:2]

        x, y = (w / 2) - 32, (h / 2) - 32

        discriminator_batch[i] = image[x:x + 64, y:y + 64, :]

    return discriminator_batch


def next_fake_batch():
    """
    Return random seeds for the generator.
    """
    batch = np.random.uniform(
        -1.0,
        1.0,
        size=[FLAGS.batch_size, FLAGS.seed_size])

    return batch.astype(np.float32)


def generate():
    """
    """
    print('generating')

    checkpoint_source_path = tf.train.latest_checkpoint(
        FLAGS.checkpoints_dir_path)

    gan_graph = build_xwgan()
    summaries = build_summaries(gan_graph)

    fake_sources = np.zeros(
        [FLAGS.summary_row_size * FLAGS.summary_col_size, FLAGS.seed_size])

    seed_o = np.random.uniform(-0.5, 0.5, size=[FLAGS.seed_size])
    seed_x = np.random.uniform(-0.1, 0.1, size=[FLAGS.seed_size])
    seed_y = np.random.uniform(-0.1, 0.1, size=[FLAGS.seed_size])

    for y in range(FLAGS.summary_row_size):
        seed_t = seed_o + y * seed_y

        for x in range(FLAGS.summary_col_size):
            fake_sources[x * FLAGS.summary_row_size + y] = seed_t + x * seed_x

    with tf.Session() as session:
        tf.train.Saver().restore(session, checkpoint_source_path)

        fetch = summaries['generated_png']

        feeds = {gan_graph['seed']: fake_sources}

        png = session.run(fetch, feed_dict=feeds)

        png_path = os.path.join(
            FLAGS.outputs_dir_path, time.strftime('%Y%m%d_%H%M%S') + '.png')

        with tf.gfile.GFile(png_path, 'wb') as f:
            f.write(png)


def train():
    """
    """
    reader = ore.RandomReader(ore.DATASET_LSUN_BEDROOM_TRAINING)

    # tensorflow
    checkpoint_source_path = tf.train.latest_checkpoint(
        FLAGS.checkpoints_dir_path)
    checkpoint_target_path = os.path.join(
        FLAGS.checkpoints_dir_path, 'model.ckpt')

    gan_graph = build_xwgan()
    summaries = build_summaries(gan_graph)

    reporter = tf.summary.FileWriter(FLAGS.logs_dir_path)

    with tf.Session() as session:
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
            for _ in range(5):
                real_sources = next_real_batch(reader)
                fake_sources = next_fake_batch()

                fetches = [
                    gan_graph['discriminator_loss'],
                    gan_graph['discriminator_trainer'],
                    summaries['discriminator_loss_summary']
                ]

                feeds = {
                    gan_graph['seed']: fake_sources,
                    gan_graph['real']: real_sources,
                }

                returns = session.run(fetches, feed_dict=feeds)

                d_loss_summary = returns[2]

            reporter.add_summary(d_loss_summary, global_step)

            #
            fake_sources = next_fake_batch()

            log_fakes = (global_step % 500 == 0)

            fetches = [
                gan_graph['global_step'],
                gan_graph['generator_loss'],
                gan_graph['generator_trainer'],
                summaries['generator_loss_summary'],
            ]

            feeds = {gan_graph['seed']: fake_sources}

            if log_fakes:
                fetches.append(summaries['generator_fake_summary'])

            returns = session.run(fetches, feed_dict=feeds)

            global_step = returns[0]
            g_loss_summary = returns[3]

            reporter.add_summary(g_loss_summary, global_step)

            if log_fakes:
                reporter.add_summary(returns[4], global_step)

            if global_step % 100 == 0:
                print('[{}]: {}'.format(global_step, returns[1]))

            if global_step % 500 == 0:
                tf.train.Saver().save(
                    session,
                    checkpoint_target_path,
                    global_step=gan_graph['global_step'])


def main(_):
    """
    """
    sanity_check()

    if FLAGS.generate:
        generate()
    else:
        train()


if __name__ == '__main__':
    """
    """
    tf.app.run()
