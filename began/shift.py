"""
"""
import began
import numpy as np
import tensorflow as tf

from six.moves import range


FLAGS = tf.app.flags.FLAGS


def load_vectors():
    """
    Load 4 vectors specified by parameters. Use uniform random vectors if the
    path is invalid.
    """
    vectors = []

    for i in range(4):
        path = FLAGS.__getattr__('vector_{}_path'.format(i))

        try:
            v = np.load(path)['v']
        except:
            v = np.random.uniform(-1.0, +1.0, size=[FLAGS.seed_size])

        vectors.append(v)

    return vectors


def build_image_grid(image_batch, row, col):
    """
    Build an image grid from an image batch.
    """
    image_size = FLAGS.image_size

    grid = tf.reshape(
        image_batch, [1, row * col * image_size, image_size, 3])
    grid = tf.split(grid, col, axis=1)
    grid = tf.concat(grid, axis=2)
    grid = tf.saturate_cast(grid * 127.5 + 127.5, tf.uint8)
    grid = tf.reshape(grid, [row * image_size, col * image_size, 3])

    return grid


def save_image_grid(network, image_grid):
    """
    Save the image grid.
    """
    variables = \
        network['generator_variables'] + network['discriminator_variables']

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        tf.train.Saver(var_list=variables) \
                .restore(session, FLAGS.checkpoint_path)

        result = session.run(image_grid)

    with tf.gfile.GFile(FLAGS.result_path, 'wb') as f:
        f.write(result)


def build_random_seeds():
    """
    Build a random batch tensors for the generator.
    """
    seeds = np.random.uniform(
        -1.0,
        +1.0,
        size=[FLAGS.experiment_size * FLAGS.experiment_size, FLAGS.seed_size])

    return tf.constant(seeds, dtype=tf.float32, name='seeds')


def build_bilinear_interp_seeds(row, col, vtl, vtr, vbl, vbr):
    """
    Build a vector grid interpolated from 4 vectors. The grid is reshaped and
    returned.
    """
    vs = np.zeros((row * col, FLAGS.seed_size))

    for i in range(row * col):
        x, y = i / row, i % row

        wr = 0.0 if col == 1 else float(x) / float(col - 1)
        wb = 0.0 if row == 1 else float(y) / float(row - 1)

        wl, wt = 1.0 - wr, 1.0 - wb

        vs[i] = wt * (wl * vtl + wr * vtr) + wb * (wl * vbl + wr * vbr)

    return tf.constant(vs, dtype=tf.float32, name='seeds')


def exp_interp_linear():
    """
    Generate an image grid.

    2 vectors are either loaded (if specified) or randomly generated. The 2
    vectors are then positioned at the 2 ends of a line. All the other
    vectors for the other cells are linear interpolated with those 2 vectors.

    The vector line is reshape into a batch and fed to the generator network.
    There are 2 rows for the final image grid. The images on the first row are
    the output of the discriminator. The images on the second row are the
    output of the generator with respect to the 1st row.
    """
    image_size = FLAGS.image_size

    vs = load_vectors()

    seeds = build_bilinear_interp_seeds(
        1, FLAGS.experiment_size, vs[0], vs[1], vs[0], vs[1])

    network = began.build_generating_network(seeds)

    image_grid_fake = build_image_grid(
        network['fake'], 1, FLAGS.experiment_size)

    image_grid_cute = build_image_grid(
        network['ae_output_fake'], 1, FLAGS.experiment_size)

    image_grid = tf.concat([image_grid_cute, image_grid_fake], axis=0)

    image_grid = tf.reshape(
        image_grid, [2 * image_size, FLAGS.experiment_size * image_size, 3])

    image_grid = tf.image.encode_png(image_grid)

    save_image_grid(network, image_grid)


def exp_interp_bilinear():
    """
    Generate an image grid.

    4 vectors are either loaded (if specified) or randomly generated. The 4
    vectors are then positioned at the 4 corners of the grid. All the other
    vectors for the other cells are bilinear interpolated with those 4 vectors.

    The vector grid is reshape into a batch and fed to the generator network.
    The final image grid is generated from the result of the discriminator.
    """
    vs = load_vectors()

    seeds = build_bilinear_interp_seeds(
        FLAGS.experiment_size,
        FLAGS.experiment_size,
        vs[0], vs[1], vs[2], vs[3])

    network = began.build_generating_network(seeds)

    image_grid_cute = build_image_grid(
        network['ae_output_fake'],
        FLAGS.experiment_size,
        FLAGS.experiment_size)

    image_grid = tf.image.encode_png(image_grid_cute)

    save_image_grid(network, image_grid)


def exp_random():
    """
    Generate an image grid with random z(s).
    """
    seeds = build_random_seeds()

    network = began.build_generating_network(seeds)

    image_grid = build_image_grid(
        network['ae_output_fake'],
        FLAGS.experiment_size,
        FLAGS.experiment_size)

    image_grid = tf.image.encode_png(image_grid)

    save_image_grid(network, image_grid)


def main(_):
    """
    """
    began.sanity_check()

    if FLAGS.experiment == 'interp_linear':
        exp_interp_linear()
    if FLAGS.experiment == 'interp_bilinear':
        exp_interp_bilinear()
    elif FLAGS.experiment == 'random':
        exp_random()


if __name__ == '__main__':
    """
    """
    tf.app.flags.DEFINE_string('experiment', 'interp_linear', '')
    tf.app.flags.DEFINE_string('checkpoint-path', '', '')
    tf.app.flags.DEFINE_string('vector-0-path', '', '')
    tf.app.flags.DEFINE_string('vector-1-path', '', '')
    tf.app.flags.DEFINE_string('vector-2-path', '', '')
    tf.app.flags.DEFINE_string('vector-3-path', '', '')
    tf.app.flags.DEFINE_string('result-path', '', '')

    tf.app.flags.DEFINE_integer('experiment-size', 8, '')

    tf.app.run()
