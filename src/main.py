"""
@author: Shalom Yiblet
"""
import numpy as np
import tensorflow as tf


def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding="SAME"):
    """max-pooling"""
    return tf.nn.max_pool(x, ksize=[1, kHeight, kWidth, 1],
                          strides=[1, strideX, strideY, 1], padding=padding, name=name)


def dropout(x, keep_pro, name=None):
    """dropout"""
    return tf.nn.dropout(x, keep_pro, name)


def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[inputD, outputD], dtype="float")
        b = tf.get_variable("b", [outputD], dtype="float")
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out


def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding="SAME"):
    """convlutional"""
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[kHeight, kWidth, channel, featureNum])
        b = tf.get_variable("b", shape=[featureNum])
        featureMap = tf.nn.conv2d(
            x, w, strides=[1, strideY, strideX, 1], padding=padding)
        out = tf.nn.bias_add(featureMap, b)
        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name=scope.name)
