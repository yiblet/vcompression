from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import types


def print_wrapper(name, func, close=True, line_width=80):
    half_wdith = line_width // 2
    n = len(name)
    first = n // 2
    last = first + (n % 2)

    print('-' * (half_wdith - first) + name + ('-' * (half_wdith - last)))
    func()
    if close:
        print('-' * line_width)


# --- plotting


def plot_codes(ax, codes, labels):
    ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
    ax.set_aspect('equal')
    ax.set_xlim(codes.min() - .1, codes.max() + .1)
    ax.set_ylim(codes.min() - .1, codes.max() + .1)
    ax.tick_params(
        axis='both',
        which='both',
        left='off',
        bottom='off',
        labelleft='off',
        labelbottom='off'
    )


def plot_samples(ax, samples):
    for index, sample in enumerate(samples):
        ax[index].imshow(sample, cmap='gray')
        ax[index].axis('off')


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    samples = samples.reshape((-1, 28, 28))

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample, cmap='Greys_r')

    return fig


# -- general utility functions


def get(dictionary, *args):
    return [dictionary[arg] for arg in args]


def count_parameters(scope=None):
    return np.sum([
        np.prod(v.get_shape().as_list())
        for v in tf.trainable_variables(scope=scope)
    ])


def NAC(x, shape):
    with tf.variable_scope('nac'):
        w_hat = tf.get_variable("w_hat", shape)
        m_hat = tf.get_variable("h_hat", shape)
        w = tf.nn.tanh(w_hat) * tf.nn.sigmoid(m_hat)
        return (tf.matmul(x, w), w)
