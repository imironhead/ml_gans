"""
"""
import began
import glob
import os
import tensorflow as tf


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

    # the input batch (uniform z) for the generator.
    seed = tf.random_uniform(
        shape=[FLAGS.batch_size, FLAGS.seed_size], minval=-1.0, maxval=1.0)

    # the input batch (real data) for the discriminator.
    real = build_dataset_reader()

    gan_graph = began.build_began(seed, real)
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
    began.sanity_check()

    sanity_check()
    train()


if __name__ == '__main__':
    """
    """
    tf.app.flags.DEFINE_string('portraits-dir-path', '', '')
    tf.app.flags.DEFINE_string('logs-dir-path', '', '')
    tf.app.flags.DEFINE_string('checkpoints-dir-path', '', '')

    tf.app.run()
