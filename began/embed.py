"""
"""
import began
import numpy as np
import tensorflow as tf

from six.moves import range

FLAGS = tf.app.flags.FLAGS


def sanity_check():
    """
    """


def load_target_image():
    """
    """
    file_names = tf.train.string_input_producer([FLAGS.target_image_path])

    _, image = tf.WholeFileReader().read(file_names)

    # Decode byte data, no gif please.
    # NOTE: tf.image.decode_image can decode both jpeg and png. However, the
    #       shape (height and width) is unknown.
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_images(image, [FLAGS.image_size, FLAGS.image_size])
    image = tf.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, 3])
    image = image / 127.5 - 1.0

    with tf.Session() as session:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        image = session.run(image)

        coord.request_stop()
        coord.join(threads)

    return tf.constant(image, name='target_image')


def build_network():
    """
    """
    seed = tf.get_variable(
        'seed',
        [1, FLAGS.seed_size],
        initializer=tf.random_uniform_initializer(-0.1, 1.0))

    real = load_target_image()

    return began.build_embed_network(seed, real)


def build_summaries(network):
    """
    """
    summaries = {}

    real = network['real']
    fake = network['fake']
    cute = network['ae_output_fake']

    image = tf.concat([real, fake, cute], axis=0)

    grid = tf.reshape(image, [1, 3 * FLAGS.image_size, FLAGS.image_size, 3])
    grid = tf.split(grid, 3, axis=1)
    grid = tf.concat(grid, axis=2)
    grid = tf.saturate_cast(grid * 127.5 + 127.5, tf.uint8)

    summaries['comparison'] = tf.summary.image('comp', grid, max_outputs=4)

    return summaries


def embed():
    """
    """
    network = build_network()
    summaries = build_summaries(network)
    reporter = tf.summary.FileWriter(FLAGS.logs_dir_path)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        variables = []

        for variable in tf.trainable_variables():
            if variable.name.startswith('d_'):
                variables.append(variable)
            elif variable.name.startswith('g_'):
                variables.append(variable)

        tf.train.Saver(var_list=variables) \
                .restore(session, FLAGS.checkpoint_path)

        for step in range(100000000):
            fetches = [
                network['loss'],
                network['seed'],
                network['trainer'],
            ]

            if step % 500 == 0:
                fetches.append(summaries['comparison'])

            returns = session.run(fetches)

            for summary in returns[3:]:
                reporter.add_summary(summary, step)

            if step % 500:
                print('[{}]: {}'.format(step, returns[0]))

                np.savez(FLAGS.result_embed_path, v=returns[1][0])


def main(_):
    """
    """
    began.sanity_check()

    sanity_check()
    embed()


if __name__ == '__main__':
    """
    """
    tf.app.flags.DEFINE_string('logs-dir-path', '', '')
    tf.app.flags.DEFINE_string('checkpoint-path', '', '')
    tf.app.flags.DEFINE_string('target-image-path', '', '')
    tf.app.flags.DEFINE_string('result-embed-path', '', '')

    tf.app.run()
