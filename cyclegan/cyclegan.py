"""
build_history: gx_history, fy_history

build_reader: x_images, y_images

"""
import os
import random
import tensorflow as tf

from six.moves import range

from model import build_cycle_gan


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'logs-dir-path', None, 'path to the directory for logs')
tf.app.flags.DEFINE_string(
    'ckpt-dir-path', None, 'path to the directory for checkpoint')
tf.app.flags.DEFINE_string(
    'gx-history-dir-path', None, 'path to the dir for caching g(x)')
tf.app.flags.DEFINE_string(
    'fy-history-dir-path', None, 'path to the dir for caching f(y)')
tf.app.flags.DEFINE_string(
    'x-images-dir-path', None, 'path to the dir for image x')
tf.app.flags.DEFINE_string(
    'y-images-dir-path', None, 'path to the dir for image y')

tf.app.flags.DEFINE_boolean(
    'is-training', True, 'build and train the model')

tf.app.flags.DEFINE_integer(
    'batch-size', 1, 'batch size for training')
tf.app.flags.DEFINE_integer(
    'history-size', 50, 'history size for training discriminator')


def build_history(dir_path, generator, num):
    """
    generate 50 images with generator into dir_path, as history
    """
    paths_wcard = os.path.join(dir_path, '*.jpg')

    paths_image = tf.gfile.Glob(paths_wcard)

    if len(paths_image) == num:
        # history is already there!
        return

    assert len(paths_image) == 0, 'history is cropped'

    # assume batch size is 1 (as the paper)
    generator = generator[0]

    history = tf.saturate_cast(generator * 127.5 + 127.5, tf.uint8)

    history = tf.image.encode_jpeg(history)

    with tf.Session() as session:
        # session.run(tf.global_variables_initializer())
        # session.run(tf.local_variables_initializer())

        # make dataset reader work
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(num):
            path = os.path.join(dir_path, '{:0>2}.jpg'.format(i))

            jpg = tf.write_file(path, history)

            session.run(jpg)

        coord.request_stop()
        coord.join(threads)


def build_batch_reader(paths_image, batch_size):
    """
    """
    file_name_queue = tf.train.string_input_producer(paths_image)

    reader_key, reader_val = tf.WholeFileReader().read(file_name_queue)

    # decode a raw input image
    image = tf.image.decode_jpeg(reader_val, channels=3)

    # random horizontal flipping to increase training data
    image = tf.image.random_flip_left_right(image)

    # to float32 and -1.0 ~ +1.0
    image = tf.cast(image, dtype=tf.float32) / 127.5 - 1.0

    # resize to 256 x 256 for the model.
    # also, a batch need concreate image size
    image = tf.image.resize_images(image, [256, 256])

    # create bacth
    return tf.train.batch(
        tensors=[image],
        batch_size=batch_size,
        capacity=batch_size * 8)


def build_image_batch_reader(dir_path, batch_size):
    """
    """
    paths_wcard = os.path.join(dir_path, '*.jpg')

    # paths_image = tf.train.match_filenames_once(paths_wcard)
    paths_image = tf.gfile.Glob(paths_wcard)

    return build_batch_reader(paths_image, batch_size)


def build_history_batch_reader(dir_path, history_size, batch_size):
    """
    """
    paths_image = []

    for i in range(history_size):
        path = os.path.join(dir_path, '{:0>2}.jpg'.format(i))

        paths_image.append(path)

    return build_batch_reader(paths_image, batch_size)


def build_history_saver(image, history_path):
    """
    """
    image = tf.saturate_cast(image * 127.5 + 127.5, tf.uint8)

    image = tf.image.encode_jpeg(image[0])

    path = os.path.join(history_path, 'temp_history.jpg')

    return tf.write_file(path, image)


def build_summaries(model):
    """
    """
    summaries = {}

    generations = [
        ('summary_x_gx', 'x_images', 'gx_images'),
        ('summary_y_fy', 'y_images', 'fy_images')]

    for g in generations:
        images = tf.concat([model[g[1]], model[g[2]]], axis=2)

        images = tf.reshape(images, [1, FLAGS.batch_size * 256, 512, 3])

        images = tf.saturate_cast(images * 127.5 + 127.5, tf.uint8)

        summary = tf.summary.image(g[0], images, max_outputs=4)

        summaries[g[0]] = summary

    return summaries


def update_history(step, history_size, history_dir_path):
    """
    """
    name_past = '{:0>2}.jpg'.format(random.randint(0, history_size - 1))
    path_temp = os.path.join(history_dir_path, 'temp_history.jpg')
    path_past = os.path.join(history_dir_path, name_past)

    tf.gfile.Rename(path_temp, path_past, True)


def train_one_step(model, summaries, ckpt_target_path, reporter):
    """
    """
    session = tf.get_default_session()

    # train discriminator for x with history
    step, _ = session.run([model['step'], model['dx_trainer']])

    # train discriminator for y with history
    session.run([model['dy_trainer']])

    # train generator g(x) and save y history
    fetches = {k: model[k] for k in ['gx_trainer', 'gx_history_saver']}

    if step % 100 == 0:
        fetches['summary_x_gx'] = summaries['summary_x_gx']

    fetched = session.run(fetches)

    if 'summary_x_gx' in fetched:
        reporter.add_summary(fetched['summary_x_gx'], step)

    # train generator f(y) and save x history
    fetches = {k: model[k] for k in ['fy_trainer', 'fy_history_saver']}

    if step % 100 == 0:
        fetches['summary_y_fy'] = summaries['summary_y_fy']

    fetched = session.run(fetches)

    if 'summary_y_fy' in fetched:
        reporter.add_summary(fetched['summary_y_fy'], step)

    #
    update_history(step, FLAGS.history_size, FLAGS.gx_history_dir_path)
    update_history(step, FLAGS.history_size, FLAGS.fy_history_dir_path)

    if step % 1000 == 0:
        tf.train.Saver().save(
            session, ckpt_target_path, global_step=model['step'])

    return step


def train():
    """
    """
    ckpt_source_path = tf.train.latest_checkpoint(FLAGS.ckpt_dir_path)
    ckpt_target_path = os.path.join(FLAGS.ckpt_dir_path, 'model.ckpt')

    x_images = build_image_batch_reader(
        FLAGS.x_images_dir_path, FLAGS.batch_size)

    y_images = build_image_batch_reader(
        FLAGS.y_images_dir_path, FLAGS.batch_size)

    gx_history = build_history_batch_reader(
        FLAGS.gx_history_dir_path, FLAGS.history_size, FLAGS.batch_size)

    fy_history = build_history_batch_reader(
        FLAGS.fy_history_dir_path, FLAGS.history_size, FLAGS.batch_size)

    model = build_cycle_gan(x_images, y_images, gx_history, fy_history, True)

    # initialize the model before build_history
    # with tf.Session() as session:
    #     session.run(tf.global_variables_initializer())
    #     session.run(tf.local_variables_initializer())
    #
    #     if ckpt_source_path is not None:
    #         tf.train.Saver().restore(session, ckpt_source_path)
    #
    # build_history(
    #     FLAGS.gx_history_dir_path, model['gx_images'], FLAGS.history_size)
    #
    # build_history(
    #     FLAGS.fy_history_dir_path, model['fy_images'], FLAGS.history_size)

    summaries = build_summaries(model)

    model['gx_history_saver'] = \
        build_history_saver(model['gx_images'], FLAGS.gx_history_dir_path)

    model['fy_history_saver'] = \
        build_history_saver(model['fy_images'], FLAGS.fy_history_dir_path)

    reporter = tf.summary.FileWriter(FLAGS.logs_dir_path)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

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
            step = train_one_step(model, summaries, ckpt_target_path, reporter)

            print('step: {}'.format(step))

        coord.request_stop()
        coord.join(threads)


def main(_):
    """
    """
    FLAGS.logs_dir_path = './logs/'
    FLAGS.ckpt_dir_path = './ckpt/'
    FLAGS.gx_history_dir_path = './gx_history/'
    FLAGS.fy_history_dir_path = './fy_history/'
    FLAGS.x_images_dir_path = '/home/ironhead/datasets/cyclegan/horse2zebra/trainA/'
    FLAGS.y_images_dir_path = '/home/ironhead/datasets/cyclegan/horse2zebra/trainB/'

    train()


if __name__ == '__main__':
    tf.app.run()
