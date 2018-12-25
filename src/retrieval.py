from __future__ import absolute_import
from global_variables import *
import pickle
import numpy as np


def cifar_input_fn(test=False):
    """Read CIFAR input data from a TFRecord dataset."""
    batch_size = FLAGS.batch_size

    def parser(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                "image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64),
            }
        )
        image = tf.decode_raw(features["image"], tf.uint8)
        image.set_shape([3 * 32 * 32])
        image = tf.cast(image, tf.float32) * (1. / 255)
        image = tf.transpose(tf.reshape(image, [3, 32, 32]))
        image = tf.image.rot90(image, 3)
        return image, image

    if test:
        location = FLAGS.tf_records_dir + '/eval.tfrecords'
    else:
        location = FLAGS.tf_records_dir + '/train.tfrecords'

    dataset = tf.data.TFRecordDataset([location])
    dataset = dataset.map(
        parser, num_parallel_calls=batch_size
    ).cache().repeat()
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(100)
    return dataset


def large_image_input_fn(test=False):
    if not FLAGS.local:
        dataset = tf.data.TextLineDataset('{FLAGS.bucket}/image_locations.txt')
    else:
        dataset = tf.io.matching_files(f'{FLAGS.large_image_dir}/**/*jpg')

    if test:
        dataset = dataset.take(FLAGS.holdout_size)
    else:
        dataset = dataset.skip(FLAGS.holdout_size)

    def parse_image(img_loc):
        image_file = tf.io.read_file(img_loc)
        image = tf.image.decode_jpeg(image_file, channels=3)
        return image

    def is_large_image(image):
        shape = tf.shape(image)[:2]
        return tf.reduce_all(shape >= FLAGS.crop_size)

    def reshape(image):
        image = tf.image.random_crop(
            image, [FLAGS.crop_size, FLAGS.crop_size, 3]
        )
        image = tf.dtypes.cast(image, tf.float32) / 255.0
        return image, image

    return dataset.shuffle(10000).repeat(
    ).map(parse_image).filter(is_large_image).map(reshape).batch(
        FLAGS.batch_size
    ).prefetch(4)


def unpickle(file):
    with open(f'{FLAGS.data}/{file}', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def parse_cifar(file):
    data = unpickle(file)
    data = data[b'data']

    image_depth = 3
    image_height = 32
    image_width = 32

    data = data.reshape([-1, image_depth, image_height, image_width])
    data = data.transpose([0, 2, 3, 1])
    data = data.astype(np.float32)
    return data / 255.0


def load_data():
    x_input = [parse_cifar(f'data_batch_{i}') for i in range(1, 6)]
    x_input = np.array(x_input).reshape([-1, *DIM])
    test_input = parse_cifar('test_batch')
    return (x_input, test_input)
