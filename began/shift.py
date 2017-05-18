"""
"""
import began
import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS


def interpolate(va, vb):
    """
    """
    vs = np.zeros((16, va.shape[0]))
    vd = (vb - va) / 15.0

    for i in range(16):
        vs[i] = va + float(i) * vd

    seeds = tf.constant(vs, dtype=tf.float32, name='seeds')

    network = began.build_generating_network(seeds)

    grid_a = tf.reshape(
        network['ae_output_fake'], [1, 16 * 128, 128, 3])
    grid_a = tf.split(grid_a, 16, axis=1)
    grid_a = tf.concat(grid_a, axis=2)

    grid_b = tf.reshape(
        network['fake'], [1, 16 * 128, 128, 3])
    grid_b = tf.split(grid_b, 16, axis=1)
    grid_b = tf.concat(grid_b, axis=2)

    final = tf.concat([grid_a, grid_b], axis=0)

    final = tf.saturate_cast(final * 127.5 + 127.5, tf.uint8)

    final = tf.reshape(final, [2 * 128, 16 * 128, 3])

    final = tf.image.encode_png(final)

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

        result = session.run(final)

    with tf.gfile.GFile(FLAGS.result_path, 'wb') as f:
        f.write(result)


def exp_interpolate():
    """
    """
    va = np.load(FLAGS.vector_a_path)['v']
    vb = np.load(FLAGS.vector_b_path)['v']

    # va = np.random.uniform(-0.5, 0.5, size=[FLAGS.seed_size])
    # vb = np.random.uniform(-0.5, 0.5, size=[FLAGS.seed_size])

    interpolate(va, vb)


def main(_):
    """
    """
    began.sanity_check()

    if FLAGS.experiment == 'interpolate':
        exp_interpolate()


if __name__ == '__main__':
    """
    """
    tf.app.flags.DEFINE_string('experiment', 'interpolate', '')
    tf.app.flags.DEFINE_string('checkpoint-path', '', '')
    tf.app.flags.DEFINE_string('vector-a-path', '', '')
    tf.app.flags.DEFINE_string('vector-b-path', '', '')
    tf.app.flags.DEFINE_string('vector-c-path', '', '')
    tf.app.flags.DEFINE_string('result-path', '', '')

    tf.app.run()
