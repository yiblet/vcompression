import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

tf.enable_eager_execution()

constant = np.arange(0, 64, 1)
constant = constant.reshape((1, 8, 8, 1)).astype(np.float)

input = tf.constant(constant, dtype=tf.float32)


def convolute(input):
    conv2d = tf.layers.Conv2D(1, [2, 2], name='test')
    output = conv2d(input)
    return output


convolute = tf.make_template('convolute', convolute)

print(convolute(input))

constant = np.arange(0, 36, 1)
constant = constant.reshape((1, 6, 6, 1)).astype(np.float)
input = tf.constant(constant, dtype=tf.float32)

print(convolute(input))
