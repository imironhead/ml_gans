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

    if FLAGS.crop_image:
        image = tf.image.crop_to_bounding_box(
            image,
            FLAGS.crop_image_offset_y,
            FLAGS.crop_image_offset_x,
            FLAGS.crop_image_size_m,
            FLAGS.crop_image_size_m)

        image = tf.random_crop(
            image, size=[FLAGS.crop_image_size_n, FLAGS.crop_image_size_n, 3])

    image = tf.image.resize_images(image, [FLAGS.image_size, FLAGS.image_size])

    image = tf.image.random_flip_left_right(image)

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

    # build generator summary
    summaries['generator'] = \
        tf.summary.scalar('generator loss', gan['generator_loss'])

    # build discriminator summaries
    d_summaries = []

    scalar_table = [
        ('convergence_measure', 'convergence measure'),
        ('discriminator_loss', 'discriminator loss'),
        ('learning_rate', 'learning rate'),
    ]

    for scalar in scalar_table:
        d_summaries.append(tf.summary.scalar(scalar[1], gan[scalar[0]]))

    summaries['discriminator_part'] = tf.summary.merge(d_summaries)

    # build image summaries
    image_table = [
        ('real', 'real image'),
        ('fake', 'generated image'),
        ('ae_output_real', 'autoencoder real'),
        ('ae_output_fake', 'autoencoder fake')
    ]

    for table in image_table:
        grid = reshape_batch_images(gan[table[0]])

        d_summaries.append(tf.summary.image(table[1], grid, max_outputs=4))

    summaries['discriminator_plus'] = tf.summary.merge(d_summaries)

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
        if checkpoint_source_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, checkpoint_source_path)

        # give up overlapped old data
        global_step = session.run(gan_graph['global_step'])

        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START),
            global_step=global_step)

        # make dataset reader work
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while True:
            # discriminator
            fetches = {
                'temp_0': gan_graph['next_k'],
                'temp_1': gan_graph['discriminator_trainer'],
            }

            if global_step % 500 == 0:
                fetches['summary'] = summaries['discriminator_plus']
            else:
                fetches['summary'] = summaries['discriminator_part']

            fetched = session.run(fetches)

            reporter.add_summary(fetched['summary'], global_step)

            # generator
            fetches = {
                'global_step': gan_graph['global_step'],
                'temp_0': gan_graph['generator_trainer'],
                'summary': summaries['generator'],
            }

            fetched = session.run(fetches)

            global_step = fetched['global_step']

            reporter.add_summary(fetched['summary'], global_step)

            if global_step % 70000 == 0:
                session.run(gan_graph['decay_learning_rate'])

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

    tf.app.flags.DEFINE_boolean('crop-image', False, '')
    tf.app.flags.DEFINE_integer('crop-image-offset-x', 25, '')
    tf.app.flags.DEFINE_integer('crop-image-offset-y', 50, '')
    tf.app.flags.DEFINE_integer('crop-image-size-m', 128, '')
    tf.app.flags.DEFINE_integer('crop-image-size-n', 128, '')

    tf.app.flags.DEFINE_integer('summary-row-size', 4, '')
    tf.app.flags.DEFINE_integer('summary-col-size', 4, '')

    # arXiv:1703.10717
    # we typically used a batch size of n = 16.
    tf.app.flags.DEFINE_integer('batch-size', 16, '')

    tf.app.run()
