"""
"""
import numpy as np
import os
import scipy.misc as misc
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'hi-images-dir-path', None, 'path to the dir of images for upper part')
tf.app.flags.DEFINE_string(
    'lo-images-dir-path', None, 'path to the dir of images for lower part')
tf.app.flags.DEFINE_string(
    'hl-images-dir-path', None, 'path to the dir for keeping result')


def concat(path_image_hi, path_image_lo, path_image_hl):
    """
    concat the upper and lower part images to the result one.
    """
    image_hi = misc.imread(path_image_hi)
    image_lo = misc.imread(path_image_lo)
    image_hl = np.concatenate([image_hi, image_lo], axis=0)

    misc.imsave(path_image_hl, image_hl)


def enumerate_images(
        path_images_dir_hi, path_images_dir_lo, path_images_dir_hl):
    """
    enumerate all paired images
    """
    hi_lo_hl_paths = []

    names_image_hi = os.listdir(path_images_dir_hi)
    names_image_lo = os.listdir(path_images_dir_lo)

    # assume images in path_images_dir_hi and path_images_dir_lo are paired
    if len(names_image_hi) != len(names_image_lo):
        raise Exception('number of images from 2 dirs are not the same')

    names_image_hi.sort()
    names_image_lo.sort()

    for hi, lo in zip(names_image_hi, names_image_lo):
        hi_path = os.path.join(path_images_dir_hi, hi)
        lo_path = os.path.join(path_images_dir_lo, lo)
        hl_path = os.path.join(path_images_dir_hl, hi)

        hi_lo_hl_paths.append((hi_path, lo_path, hl_path))

    return hi_lo_hl_paths


if __name__ == '__main__':
    hi_lo_hl_paths = enumerate_images(
        FLAGS.hi_images_dir_path,
        FLAGS.lo_images_dir_path,
        FLAGS.hl_images_dir_path)

    for i, (hi, lo, hl) in enumerate(hi_lo_hl_paths):
        print '{} / {}'.format(i, len(hi_lo_hl_paths))

        concat(hi, lo, hl)
