"""
@author: Shalom Yiblet
"""
import numpy as np
import tensorflow as tf

WIDTH = 512
HEIGHT = WIDTH

SHAPE = [
    HEIGHT,
    WIDTH,
    3,
]


def get_dims(tensor):
    return [x.value for x in tf.shape(tensor)]


def get_image_dims(tensor):
    height_idx = 1
    width_idx = 2
    return get_dims(images)[height_idx:width_idx + 1]


def downsize(images, height_factor=2, width_factor=2):
    width, height = get_image_dims(images)
    return tf.image.resize_images(  # resize_images is possibly broken
        images,
        [height / height_factor, width / width_factor]
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )


def upsize(images, height_factor=2, width_factor=2):
    width, height = get_image_dims(images)
    return tf.image.resize_images(  # resize_images is possibly broken
        images,
        [height * height_factor, width * width_factor]
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )


def compress(arg):
    # TODO
    pass


def contract(images):
    if get_image_dims(images)[0] == 2:
        return [images]
    smaller_images = resize(images)
    contracted_images = contract(smaller_images)
    smaller_images = contracted_images[-1]
    compress(images - upsize_images(smaller_images))
