"""
"""
import glob
import os
import scipy.misc
import tensorflow as tf


tf.app.flags.DEFINE_string('source-dir-path', '', '')
tf.app.flags.DEFINE_string('target-dir-path', '', '')

tf.app.flags.DEFINE_integer('image-scale', 128, '')

FLAGS = tf.app.flags.FLAGS


def sanity_check():
    """
    """
    if not os.path.isdir(FLAGS.source_dir_path):
        raise Exception('invalid source dir: {}'.format(FLAGS.source_dir_path))

    if not os.path.isdir(FLAGS.target_dir_path):
        os.makedirs(FLAGS.target_dir_path)

    if FLAGS.image_scale < 2:
        raise Exception('invalid image scale: {}'.format(FLAGS.image_scale))


def crop():
    """
    """
    paths_png_wildcards = os.path.join(FLAGS.source_dir_path, '*.png')

    paths_png = glob.glob(paths_png_wildcards)

    for path_png in paths_png:
        name_png = os.path.split(path_png)[1]
        path_new = os.path.join(FLAGS.target_dir_path, name_png)

        image = scipy.misc.imread(path_png)

        image = image[50:178, 25:153, :]

        if FLAGS.image_scale != 128:
            image = scipy.misc.imresize(image, 100 * FLAGS.image_scale / 128)

        scipy.misc.imsave(path_new, image)

        print('{} done'.format(name_png))


def main(_):
    """
    """
    sanity_check()
    crop()


if __name__ == '__main__':
    tf.app.run()
