"""
"""
import glob
import os
import tensorflow as tf

from model import build_pix2pix

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('lambda-value', 500.0, '')
tf.app.flags.DEFINE_float('learning-rate', 0.0001, '')
tf.app.flags.DEFINE_string('logs-dir-path', '', '')
tf.app.flags.DEFINE_string('ckpt-dir-path', '', '')
tf.app.flags.DEFINE_string('images-path', '', '')
tf.app.flags.DEFINE_string('source-image-path', '', '')
tf.app.flags.DEFINE_string('target-image-path', '', '')
tf.app.flags.DEFINE_boolean('is-training', True, '')
tf.app.flags.DEFINE_boolean('swap-images', False, '')
tf.app.flags.DEFINE_integer('batch-size', 1, '')
tf.app.flags.DEFINE_integer('crop-image-size', 512, '')


def build_dataset_reader():
    """
    assume each image is concatenated with source and target images:
    [ image_hi ]
    [ image_lo ]

    return:
        image batch with 6 channels, [image_hi, image_lo]
    """
    if FLAGS.is_training:
    paths_wildcards = os.path.join(FLAGS.images_path, '*.jpg')

    paths_image = glob.glob(paths_wildcards)
    else:
        paths_image = [FLAGS.source_image_path]

    file_name_queue = tf.train.string_input_producer(paths_image)

    reader_key, reader_val = tf.WholeFileReader().read(file_name_queue)

    # decode a raw input image
    image = tf.image.decode_jpeg(reader_val, channels=3)

    # random horizontal flipping to increase training data
    image = tf.image.random_flip_left_right(image)

    # split to 2 sub-images
    image_hi, image_lo = tf.split(image, 2, axis=0)

    # swap source and target
    if FLAGS.swap_images:
        image_hi, image_lo = image_lo, image_hi

    # concat to a new tensor with 6 channels:
    # image_hi_r, image_hi_g, image_hi_b,
    # image_lo_r, image_lo_g, image_lo_b,
    image = tf.concat([image_hi, image_lo], axis=2)

    # random cropping to increase training data
    image = tf.random_crop(
        image, size=[FLAGS.crop_image_size, FLAGS.crop_image_size, 6])

    # to float32 and -1.0 ~ +1.0
    image = tf.cast(image, dtype=tf.float32) / 127.5 - 1.0

    # resize to 256 x 256 for the model.
    # also, a batch need concreate image size
    image = tf.image.resize_images(image, [256, 256])

    if FLAGS.is_training:
    # create bacth
    batch_tensors = tf.train.batch(
        tensors=[image],
        batch_size=FLAGS.batch_size,
            capacity=FLAGS.batch_size * 8)
    else:
        batch_tensors = tf.reshape(image, [1, 256, 256, 6])

    # split to image_source and image_target
    return tf.split(batch_tensors, 2, axis=3)


def build_summaries(model):
    """
    """
    images = []

    for k in ['source_images', 'target_images', 'output_images']:
        temp = tf.reshape(model[k], [1, FLAGS.batch_size * 256, 256, 3])

        images.append(temp)

    images = tf.concat(images, axis=2)

    images = tf.saturate_cast(images * 127.5 + 127.5, tf.uint8)

    summary = tf.summary.image('images', images, max_outputs=4)

    return {
        'summary': summary,
    }


def build_output(model):
    """
    """
    images = tf.concat(
        [model['source_images'], model['output_images']], axis=2)

    images = tf.reshape(images, [FLAGS.batch_size * 256, 512, 3])

    images = tf.saturate_cast(images * 127.5 + 127.5, tf.uint8)

    images = tf.image.encode_png(images)

    return tf.write_file(FLAGS.target_image_path, images)


def train():
    """
    """
    ckpt_source_path = tf.train.latest_checkpoint(FLAGS.ckpt_dir_path)
    ckpt_target_path = os.path.join(FLAGS.ckpt_dir_path, 'model.ckpt')

    source_images, target_images = build_dataset_reader()

    model = build_pix2pix(
        source_images,
        target_images,
        FLAGS.lambda_value,
        FLAGS.learning_rate,
        True)

    summaries = build_summaries(model)

    reporter = tf.summary.FileWriter(FLAGS.logs_dir_path)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        if ckpt_source_path is not None:
            tf.train.Saver().restore(session, ckpt_source_path)

        # give up overlapped old data
        step = session.run(model['step'])

        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START),
            global_step=step)

        # make dataset reader work
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while True:
            fetches = {
                'step': model['step'],
                'd_trainer': model['d_trainer'],
                'summary': summaries['summary'],
            }

            fetched = session.run(fetches)

            if fetched['step'] % 100 == 0:
                reporter.add_summary(fetched['summary'], fetched['step'])
                print fetched['step']

            if fetched['step'] % 10000 == 0:
                tf.train.Saver().save(
                    session, ckpt_target_path, global_step=model['step'])

            fetches = {
                'g_trainer': model['g_trainer'],
            }

            fetched = session.run(fetches)

        coord.request_stop()
        coord.join(threads)


def translate():
    """
    """
    ckpt_source_path = tf.train.latest_checkpoint(FLAGS.ckpt_dir_path)

    source_images, target_images = build_dataset_reader()

    model = build_pix2pix(
        source_images,
        target_images,
        FLAGS.lambda_value,
        FLAGS.learning_rate,
        False)

    output = build_output(model)

    with tf.Session() as session:
        tf.train.Saver().restore(session, ckpt_source_path)

        # make dataset reader work
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        session.run(output)

        coord.request_stop()
        coord.join(threads)


def main(_):
    """
    """
    if FLAGS.is_training:
        train()
    else:
        translate()


if __name__ == '__main__':
    tf.app.run()
