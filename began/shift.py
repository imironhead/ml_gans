"""
"""
import began
import numpy as np
import tensorflow as tf

from six.moves import range


FLAGS = tf.app.flags.FLAGS


def bilinear_interpolate(vtl, vtr, vbl, vbr):
    """
    """
    step = FLAGS.interpolation_step
    image_size = FLAGS.image_size
    seed_size = FLAGS.seed_size

    vs = np.zeros((step * step, seed_size))

    for i in range(step * step):
        x, y = i / step, i % step

        wr, wb = float(x) / float(step - 1), float(y) / float(step - 1)
        wl, wt = 1.0 - wr, 1.0 - wb

        vs[i] = wt * (wl * vtl + wr * vtr) + wb * (wl * vbl + wr * vbr)

    seeds = tf.constant(vs, dtype=tf.float32, name='seeds')

    network = began.build_generating_network(seeds)

    grid = tf.reshape(
        network['ae_output_fake'],
        [1, step * step * image_size, image_size, 3])
    grid = tf.split(grid, step, axis=1)
    grid = tf.concat(grid, axis=2)
    grid = tf.saturate_cast(grid * 127.5 + 127.5, tf.uint8)
    grid = tf.reshape(grid, [step * image_size, step * image_size, 3])
    grid = tf.image.encode_png(grid)

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

        result = session.run(grid)

    with tf.gfile.GFile(FLAGS.result_path, 'wb') as f:
        f.write(result)


def interpolate(va, vb):
    """
    """
    step = FLAGS.interpolation_step

    vs = np.zeros((step, va.shape[0]))
    vd = (vb - va) / float(step - 1)

    for i in range(step):
        vs[i] = va + float(i) * vd

    seeds = tf.constant(vs, dtype=tf.float32, name='seeds')

    network = began.build_generating_network(seeds)

    grid_a = tf.reshape(
        network['ae_output_fake'], [1, step * 128, 128, 3])
    grid_a = tf.split(grid_a, step, axis=1)
    grid_a = tf.concat(grid_a, axis=2)

    grid_b = tf.reshape(
        network['fake'], [1, step * 128, 128, 3])
    grid_b = tf.split(grid_b, step, axis=1)
    grid_b = tf.concat(grid_b, axis=2)

    final = tf.concat([grid_a, grid_b], axis=0)

    final = tf.saturate_cast(final * 127.5 + 127.5, tf.uint8)

    final = tf.reshape(final, [2 * 128, step * 128, 3])

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

    interpolate(va, vb)


def exp_random():
    """
    """
    vs = [np.random.uniform(-0.95, 0.95, size=[FLAGS.seed_size])
          for _ in range(4)]

    bilinear_interpolate(*vs)


def main(_):
    """
    """
    began.sanity_check()

    if FLAGS.experiment == 'interpolate':
        exp_interpolate()
    elif FLAGS.experiment == 'random':
        exp_random()


if __name__ == '__main__':
    """
    """
    tf.app.flags.DEFINE_string('experiment', 'interpolate', '')
    tf.app.flags.DEFINE_string('checkpoint-path', '', '')
    tf.app.flags.DEFINE_string('vector-a-path', '', '')
    tf.app.flags.DEFINE_string('vector-b-path', '', '')
    tf.app.flags.DEFINE_string('vector-c-path', '', '')
    tf.app.flags.DEFINE_string('result-path', '', '')

    tf.app.flags.DEFINE_integer('interpolation-step', 8, '')

    tf.app.run()
