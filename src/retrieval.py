from __future__ import absolute_import
from global_variables import *
import pickle
import numpy as np


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
