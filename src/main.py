import os
import pprint
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

WIDTH = 32
HEIGHT = WIDTH
CHANNEL = 3

SHAPE = [
    HEIGHT,
    WIDTH,
    3,
]

LOCAL = 'COLAB_TPU_ADDR' not in os.environ


if LOCAL:
    DIRECTORY = 'out'
    DATA = 'data/cifar10'
    print('running locally')
else:
    DIRECTORY = '/gdrive/My Drive/data'
    DATA = 'cifar-10-batches-py'

    TPU_ADDRESS = 'grpc://' + os.environ['COLAB_TPU_ADDR']
    print('TPU address is', TPU_ADDRESS)

    with tf.Session(TPU_ADDRESS) as session:
        devices = session.list_devices()

    print('TPU devices:')
    pprint.pprint(devices)

tf.reset_default_graph()


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
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
    return data


def construct_vae(original_dim, hidden=16, z_dims=16, learning_rate=1e-3):
    x_input = tf.placeholder(tf.float32, shape=[None, *original_dim])
    z_input = tf.placeholder(tf.float32, shape=[None, z_dims])

    with tf.variable_scope('Q'):
        conv_1 = tf.layers.conv2d(
            x_input,
            3,
            [2, 2],
            name='conv_1',
            activation=tf.nn.relu
        )
        conv_2 = tf.layers.conv2d(
            conv_1,
            16,
            [2, 2],
            (2, 2),
            name='conv_2',
            activation=tf.nn.relu
        )
        conv_3 = tf.layers.conv2d(
            conv_2,
            16,
            [2, 2],
            name='conv_3',
            activation=tf.nn.relu
        )
        conv_4 = tf.layers.conv2d(
            conv_3,
            16,
            [2, 2],
            name='conv_4',
            activation=tf.nn.relu
        )

    out = tf.layers.flatten(conv_4)

    latent = tf.layers.dense(out, hidden, activation=tf.nn.relu)

    z_mu = tf.layers.dense(latent, z_dims, activation=None)
    z_log_var = tf.layers.dense(latent, z_dims, activation=None)
    epsilon = tf.random_normal(shape=tf.shape(
        z_log_var), mean=0, stddev=1, dtype=tf.float32)

    z = z_mu + tf.exp(z_log_var / 2.0) * epsilon

    def generator(z):
        with tf.variable_scope('P', reuse=tf.AUTO_REUSE):
            hidden_0 = tf.layers.dense(z, hidden, activation=tf.nn.relu)
            hidden_0 = tf.layers.dense(
                hidden_0, 13 * 13 * 32, activation=tf.nn.relu)
            hidden_0 = tf.reshape(hidden_0, (-1, 13, 13, 32))

            decon_1 = tf.layers.conv2d_transpose(
                hidden_0,
                16,
                [2, 2],
                strides=(1, 1),
                name='decon_1',
                activation=tf.nn.relu
            )
            decon_2 = tf.layers.conv2d_transpose(
                decon_1,
                16,
                [3, 3],
                strides=(2, 2),
                name='decon_2',
                activation=tf.nn.relu
            )
            decon_3 = tf.layers.conv2d_transpose(
                decon_2,
                16,
                [3, 3],
                strides=(1, 1),
                name='decon_3',
                activation=tf.nn.relu
            )
            decon_4 = tf.layers.conv2d_transpose(
                decon_3,
                3,
                [2, 2],
                name='decon_4',
                activation=None
            )
            return (tf.nn.sigmoid(decon_4), decon_4)

    _, x_hat = generator(z)
    x_gen, _ = generator(z_input)

    print(x_hat, x_input)

    recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_hat, labels=x_input), axis=[1, 2, 3]
    ) * WIDTH * HEIGHT * CHANNEL

    latent_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mu) - tf.exp(z_log_var), axis=1
    )

    total_loss = tf.reduce_mean(recon_loss + latent_loss)
    train_op = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(total_loss)

    return (x_input, z_input, x_gen, x_hat, total_loss, train_op)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)

    return fig


def main():

    data = [
        parse_cifar(f'{DATA}/data_batch_{i}')
        for i in range(1, 6)
    ]

    data = np.array(data)
    data = data.reshape((-1, 32, 32, 3))

    print(data.shape)

    test = parse_cifar(f'{DATA}/test_batch')
    print(test.shape)
    epochs = 1

    (x_input, z_input, x_gen, x_hat, total_loss,
     train_op) = construct_vae((32, 32, 3))

    if LOCAL:
        sess = tf.Session()
    else:
        sess = tf.Session(TPU_ADDRESS)

    data = data / 255.0

    try:
        sess.run(tf.global_variables_initializer())
        if not LOCAL:
            print('Initializing TPUs...')
            sess.run(tf.contrib.tpu.initialize_system())

        print('Running ops')

        if not os.path.exists(DIRECTORY):
            os.mkdir(DIRECTORY)

        for x in range(10000):
            X_mb = data[np.random.choice(
                data.shape[0], 1000, replace=False), ...]
            _, loss = sess.run([train_op, total_loss],
                               feed_dict={x_input: X_mb})

            if (x % 100) == 0:
                samples = sess.run(x_gen, feed_dict={
                    z_input: np.random.randn(16, 16)})
                fig = plot(samples)
                plt.title(f'Epoch {x}')
                plt.savefig(f'{DIRECTORY}/epoch_{x}.png')
                if not LOCAL:
                    plt.show()
                plt.close()

    finally:
        # For now, TPU sessions must be shutdown separately from
        # closing the session.
        sess.run(tf.contrib.tpu.shutdown_system())
        sess.close()


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
        'encoder_h1': tf.Variable(tf.truncated_normal(
            [5, 5, channels['input'], channels['h1']],
            stddev=0.01)
        ),
        'encoder_h2': tf.Variable(tf.truncated_normal(
            [5, 5, channels['h1'], channels['h2']], stddev=0.01)
        ),
        'decoder_h3': tf.Variable(tf.truncated_normal(
            [5, 5, channels['h2'], channels['h3']], stddev=0.01)
        ),
        'decoder_h4': tf.Variable(tf.truncated_normal(
            [5, 5, channels['h3'], channels['output']], stddev=0.01)
        ),
    }

    biases = {
        'encoder_h1_biases': tf.Variable(tf.truncated_normal(
            [channels['h1']],
            stddev=0.01)
        ),
        'encoder_h2_biases': tf.Variable(tf.truncated_normal(
            [channels['h2']], stddev=0.01)
        ),
        'decoder_h3_biases': tf.Variable(tf.truncated_normal(
            [channels['h3']], stddev=0.01)
        ),
        'decoder_h4_biases': tf.Variable(tf.truncated_normal(
            [channels['output']], stddev=0.01)
        ),
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


if __name__ == "__main__":
    main()
