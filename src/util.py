from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import types

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


def print_param_count(scope=None):
    if scope is not None:
        count = count_parameters(scope=f".*{scope}")
    else:
        count = count_parameters()
        scope = 'vae'
    print(f'number of parameters in {scope}: {count}')


def compress(arg):
    placeholder = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

    channels = {
        'input': 3,
        'h1': 18,
        'h2': 1,
        'h3': 6,
        'output': 3,
    }

    variables = {
        'encoder_h1':
        tf.Variable(
            tf.truncated_normal([5, 5, channels['input'], channels['h1']],
                                stddev=0.01)
        ),
        'encoder_h2':
        tf.Variable(
            tf.truncated_normal([5, 5, channels['h1'], channels['h2']],
                                stddev=0.01)
        ),
        'decoder_h3':
        tf.Variable(
            tf.truncated_normal([5, 5, channels['h2'], channels['h3']],
                                stddev=0.01)
        ),
        'decoder_h4':
        tf.Variable(
            tf.truncated_normal([5, 5, channels['h3'], channels['output']],
                                stddev=0.01)
        ),
    }

    biases = {
        'encoder_h1_biases':
        tf.Variable(tf.truncated_normal([channels['h1']], stddev=0.01)),
        'encoder_h2_biases':
        tf.Variable(tf.truncated_normal([channels['h2']], stddev=0.01)),
        'decoder_h3_biases':
        tf.Variable(tf.truncated_normal([channels['h3']], stddev=0.01)),
        'decoder_h4_biases':
        tf.Variable(tf.truncated_normal([channels['output']], stddev=0.01)),
    }

    output = model(placeholder)
    print(output)
    pass


def contract(images):
    if get_image_dims(images)[0] == 2:
        return [images]
    smaller_images = resize(images)
    contracted_images = contract(smaller_images)
    smaller_images = contracted_images[-1]
    compress(images - upsize_images(smaller_images))


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
        [height / height_factor, width / width_factor],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )


def upsize(images, height_factor=2, width_factor=2):
    width, height = get_image_dims(images)
    return tf.image.resize_images(  # resize_images is possibly broken
        images,
        [height * height_factor, width * width_factor],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )


def NAC(x, shape):
    with tf.variable_scope('nac'):
        w_hat = tf.get_variable("w_hat", shape)
        m_hat = tf.get_variable("h_hat", shape)
        w = tf.nn.tanh(w_hat) * tf.nn.sigmoid(m_hat)
        return (tf.matmul(x, w), w)
