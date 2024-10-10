import os

import tensorflow as tf


def read_image(filename, directory='', channels=3):
    """
    Read image, can be applied/mapped to tf.data.Dataset!

    :param filename: string of image to be read
    :param directory: where to read image from
    :param channels: 3 channels for RGB

    :return: image as EagerTensor
    """
    if isinstance(filename, bytes):
        filename = filename.decode("utf-8")

    if isinstance(directory, bytes):
        directory = directory.decode("utf-8")

    image = tf.io.read_file(os.path.join(directory, filename))
    try:
        image = tf.image.decode_png(image, channels=channels)
        image = tf.cast(image, dtype=tf.dtypes.float32)
    except:
        print('file not readable:', filename)

    return image


def tf_read_image(filename, directory="", channels=3, original_image_shape=2048):
    """
    tensorflow wrapper to read image
    """
    image = tf.numpy_function(read_image, (filename, directory, channels), tf.float32)
    return image
