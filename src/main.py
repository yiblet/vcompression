# @title The Big File { display-mode: "form" }
from __future__ import absolute_import
import os
import pprint
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import subprocess
import sys
import util
import layers

WIDTH = 32
HEIGHT = WIDTH
CHANNEL = 3

DIM = (HEIGHT, WIDTH, CHANNEL)

URL_LOG = 'url.txt'

DEFAULT_SUMMARY_COLLECTION = 'summaries'

FLAGS = util.Namespace(is_set=False)


def define_flags(additional_flags=None):
    if additional_flags is not None:
        FLAGS.bulk_update(additional_flags)

    reset = False  # @param {type: "boolean"}
    if (not reset) and FLAGS.is_set:
        return

    FLAGS.batch_size = 16  # @param {type: "number"}
    FLAGS.epochs = 1000  # @param {type: "number"}
    FLAGS.is_set = True
    FLAGS.learning_rate = 1e-3  # @param {type: "number"}
    FLAGS.summary_frequency = 200  # @param {type: "number"}
    FLAGS.train_steps = 600  # @param {type: "number"}
    FLAGS.z_dims = 128  # @param {type: "number"}
    FLAGS.summarize = True
    FLAGS.local = os.uname()[1] == 'XPS'

    if FLAGS.local:
        FLAGS.data = 'data/cifar10'
        FLAGS.debug = True
        FLAGS.directory = 'out'
        FLAGS.summaries_dir = 'local/summaries'
        FLAGS.tpu_address = None

        print('running locally')
    else:
        print('mounting google drive')
        from google.colab import drive
        drive.mount('/gdrive')

        FLAGS.data = '/gdrive/My Drive/cifar10'
        FLAGS.debug = False
        FLAGS.directory = '/gdrive/My Drive/data_mnist'
        summaries_dir = 'summaries'  # @param {type: "string"}
        FLAGS.summaries_dir = f'/gdrive/My Drive/{summaries_dir}'
        FLAGS.tpu_address = None

        if 'COLAB_TPU_ADDR' in os.environ:
            FLAGS.tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']

            print('TPU address is', FLAGS.tpu_address)

            with tf.Session(FLAGS.tpu_address) as session:
                devices = session.list_devices()

            print('TPU devices:')
            pprint.pprint(devices)

        subprocess.Popen(
            "kill $(ps -A | grep tensorboard | grep -o '^[0-9]\\+')",
            shell=True,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
        ) .communicate()

        subprocess.Popen(
            "kill $(ps -A | grep lt | grep -o '^[0-9]\\+')",
            shell=True,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
        ) .communicate()

        subprocess.Popen(
            f"rm '{URL_LOG}'",
            shell=True,
        )

        print(subprocess.Popen(
            f"npm install -g localtunnel; lt --port 6006 -s yiblet > {URL_LOG} 2>&1 &",
            shell=True,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
        ).communicate()[0].decode('ascii'))

        subprocess.Popen(
            f"rm -r '{FLAGS.summaries_dir}'",
            shell=True,
        ).wait()

        subprocess.Popen(
            f"tensorboard --logdir '{FLAGS.summaries_dir}' --host 0.0.0.0 >> tensorboard.log 2>&1 &",
            shell=True,
        )


tf.reset_default_graph()


def variable_summaries(key, var, collection):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    with tf.name_scope(f'{key}_summaries'):
        if not FLAGS.summarize:
            tf.summary.histogram('histogram', var)
        else:
            mean = tf.reduce_mean(var)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.scalar('stddev', stddev)
            tf.summary.histogram('histogram', var)


class SummaryScope(dict):
    def __init__(self, scope_name, collection=DEFAULT_SUMMARY_COLLECTION):
        super()
        self.scope_name = scope_name
        self.collection = collection

    def _get_name(self, name):
        name = name[:name.rindex('/')]
        name = name[name.rindex('/') + 1:]
        return name

    def sequential(self, input, ops, include_input=False):
        prev_op = input

        x = str("")

        if include_input:
            self[self._get_name(prev_op.name)] = prev_op

        for operation in ops:
            prev_op = operation(prev_op)
            name = self._get_name(prev_op.name)
            self[name] = prev_op
            new_op = True

        return prev_op

    def __enter__(self):
        self.scope = tf.name_scope(self.scope_name)
        self.scope.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.scope.__exit__(type, value, traceback)

        if FLAGS.debug:
            print(f'printing {self.scope_name} dimensions')
            for k, v in self.items():
                print(f'{k}: {v.shape}')
            print('')

        for (key, var) in self.items():
            variable_summaries(key, var, self.collection)


def unpickle(file):
    import pickle
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


def construct_vae(original_dim):
    import tensorflow_probability as tfp
    tfd = tfp.distributions

    def make_encoder(data):
        with SummaryScope('Q-probability') as scope:
            x = tf.reshape(data, (-1, *original_dim))
            x = scope.sequential(
                x,
                [
                    tf.layers.Conv2D(
                        128,
                        [2, 2],
                        [2, 2],
                        name='conv_1',
                        activation=None
                    ),
                    tf.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu', name='relu_1'),
                    layers.ResidualBlock(
                        128,
                        kernel=[3, 3],
                        activation=tf.nn.relu
                    ),
                    tf.layers.Conv2D(
                        128,
                        [2, 2],
                        [2, 2],
                        name='conv_2',
                        activation=None,
                        _reuse=True,
                    ),
                    tf.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu', name='relu_2'),
                    layers.ResidualBlock(
                        128,
                        kernel=[3, 3],
                        activation=tf.nn.relu
                    ),
                    tf.layers.Conv2D(
                        128,
                        [2, 2],
                        [2, 2],
                        name='conv_3',
                        activation=None,
                    ),
                    tf.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu', name='relu_3'),
                    layers.ResidualBlock(
                        128,
                        kernel=[3, 3],
                        activation=tf.nn.relu
                    ),
                    tf.layers.Conv2D(
                        128,
                        [2, 2],
                        [2, 2],
                        name='conv_4',
                        activation=None,
                    ),
                    tf.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu', name='relu_4'),
                    tf.layers.Conv2D(
                        128,
                        [2, 2],
                        [1, 1],
                        name='conv_5',
                        activation=tf.nn.relu,
                    ),
                ],
            )

            latent = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))

        with SummaryScope('z-probability') as scope:
            loc = tf.layers.dense(latent, FLAGS.z_dims, name='loc')
            scope['loc'] = loc
            scale = tf.layers.dense(
                latent, FLAGS.z_dims,  tf.nn.softplus, name='scale')
            scope['scale'] = scale

        return tfd.MultivariateNormalDiag(loc, scale)

    def make_prior():
        loc = tf.zeros(FLAGS.z_dims)
        scale = tf.ones(FLAGS.z_dims)
        return tfd.MultivariateNormalDiag(loc, scale)

    def make_decoder(code):
        with SummaryScope('P-probability') as scope:
            x = scope.sequential(
                tf.reshape(code, (-1, 1, 1, FLAGS.z_dims)),
                [
                    tf.layers.Conv2DTranspose(
                        128,
                        [2, 2],
                        [1, 1],
                        name='deconv_1',
                        activation=None,
                    ),
                    tf.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu', name='relu_1'),
                    tf.layers.Conv2DTranspose(
                        128,
                        [2, 2],
                        [2, 2],
                        name='deconv_2',
                        activation=None,
                    ),
                    tf.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu', name='relu_2'),
                    layers.ResidualBlock(
                        128,
                        kernel=[3, 3],
                        activation=tf.nn.relu
                    ),
                    tf.layers.Conv2DTranspose(
                        128,
                        [2, 2],
                        [2, 2],
                        name='deconv_3',
                        activation=None,
                    ),
                    tf.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu', name='relu_3'),
                    layers.ResidualBlock(
                        128,
                        kernel=[3, 3],
                        activation=tf.nn.relu
                    ),
                    tf.layers.Conv2DTranspose(
                        128,
                        [2, 2],
                        [2, 2],
                        name='deconv_4',
                        activation=None,
                    ),
                    tf.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu', name='relu_4'),
                    layers.ResidualBlock(
                        128,
                        kernel=[3, 3],
                        activation=tf.nn.relu
                    ),
                    tf.layers.Conv2DTranspose(
                        128,
                        [2, 2],
                        [2, 2],
                        name='deconv_5',
                        activation=None,
                    ),
                    tf.layers.BatchNormalization(),
                    tf.keras.layers.Activation('relu', name='relu_5'),
                    tf.layers.Conv2DTranspose(
                        3,
                        [1, 1],
                        [1, 1],
                        name='deconv_6',
                        activation=None,
                    ),
                ]
            )

        with SummaryScope('logit') as scope:
            logit = tf.reshape(x, [-1, *original_dim])
            scope['logit'] = logit

        res = tfd.Independent(tfd.Bernoulli(logit))
        return res

    data = tf.placeholder(tf.float32, [None, *original_dim])

    make_encoder = tf.make_template('encoder', make_encoder)
    make_decoder = tf.make_template('decoder', make_decoder)

    # Define the model.
    prior = make_prior()
    posterior = make_encoder(data)
    code = posterior.sample()

    # Define the loss.
    with SummaryScope('losses') as scope:
        likelihood = make_decoder(code).log_prob(data)
        scope['likelihood'] = likelihood
        divergence = tfd.kl_divergence(posterior, prior)
        scope['divergence'] = divergence
        elbo = tf.reduce_mean(likelihood - divergence)
        scope['elbo'] = elbo

    optimize = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(-elbo)

    samples = make_decoder(prior.sample(10)).mean()

    random_sub_batch_dims = tf.random_uniform(
        [10],
        minval=0,
        maxval=tf.shape(code,  out_type=tf.int32)[0],
        dtype=tf.int32,
    )

    code_random_sub_batch = tf.gather(code, random_sub_batch_dims)
    data_random_sub_batch = tf.gather(data, random_sub_batch_dims)

    generated = make_decoder(code_random_sub_batch).mean()

    merged = tf.summary.merge_all()

    sample_summary = tf.summary.image(
        'samples',
        samples,
        max_outputs=10
    )

    image_comparison = tf.concat([data_random_sub_batch, generated], 1)
    comparison_summary = tf.summary.image(
        'comparison',
        image_comparison,
        max_outputs=10
    )

    images_summary = tf.summary.merge([sample_summary, comparison_summary])

    return (data, elbo, code, samples, optimize, merged, images_summary)


def plot_codes(ax, codes, labels):
    ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
    ax.set_aspect('equal')
    ax.set_xlim(codes.min() - .1, codes.max() + .1)
    ax.set_ylim(codes.min() - .1, codes.max() + .1)
    ax.tick_params(
        axis='both', which='both', left='off', bottom='off',
        labelleft='off', labelbottom='off')


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


def count_parameters(scope=None):
    return np.sum(
        [np.prod(v.get_shape().as_list())
         for v in tf.trainable_variables(scope=scope)
         ])


def print_param_count(scope=None):
    if scope is not None:
        count = count_parameters(scope=f".*{scope}")
    else:
        count = count_parameters()
        scope = 'vae'
    print(
        f'number of parameters in {scope}: {count}'
    )


def gdn(input):
    return tf.contrib.layers.gdn(input)


def inverse_gdn(input):
    return tf.contrib.layers.gdn(input, inverse=True)


def main():
    original_dim = (HEIGHT, WIDTH, CHANNEL)

    print('-----FLAGS----')
    pprint.pprint(FLAGS.__dict__)
    print('--------------')

    (x_input, elbo, code, samples, optimize, merged, images) = construct_vae(
        original_dim
    )

    print('---------------')
    print_param_count()
    print('---------------')
    print_param_count('encoder')
    print('---------------')
    print_param_count('decoder')
    print('---------------')

    if FLAGS.debug:
        sys.exit()

    train, test = load_data()
    print(train.shape, test.shape)

    if FLAGS.tpu_address is None:
        sess = tf.Session()
    else:
        sess = tf.Session(FLAGS.tpu_address)

    if os.path.exists(FLAGS.summaries_dir):
        subprocess.run(['rm', '-r', FLAGS.summaries_dir])

    train_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/train',
        sess.graph
    )
    test_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/test'
    )

    try:
        sess.run(tf.global_variables_initializer())

        if FLAGS.tpu_address is not None:
            print('Initializing TPUs...')
            sess.run(tf.contrib.tpu.initialize_system())

        print('Running ops')

        try:
            with (open(URL_LOG, 'r')) as log:
                print(log.read())
        except FileNotFoundError:
            print('not running localtunnel')

        if not os.path.exists(FLAGS.directory):
            os.mkdir(FLAGS.directory)

        for epoch in range(FLAGS.epochs):
            feed = {
                x_input:
                test[np.random.choice(
                    test.shape[0], 100, replace=False), ...]
                .reshape([-1, *original_dim])
            }

            test_elbo, test_samples, summary, images_summary = sess.run(
                [elbo, samples, merged, images],
                feed
            )
            test_writer.add_summary(summary, FLAGS.train_steps * epoch)
            test_writer.add_summary(images_summary, FLAGS.train_steps * epoch)
            print(f'Epoch {epoch} elbo: {test_elbo}')

            # training step
            for train_step in range(FLAGS.train_steps):
                feed = {
                    x_input: train
                    [np.random.choice(
                        train.shape[0], FLAGS.batch_size, replace=False), ...]
                    .reshape([-1, *original_dim])
                }

                global_step = FLAGS.train_steps * epoch + train_step
                if global_step == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, summary = sess.run(
                        [optimize, merged],
                        feed,
                        options=run_options,
                        run_metadata=run_metadata
                    )
                    train_writer.add_summary(
                        summary, global_step)
                    train_writer.add_run_metadata(
                        run_metadata,
                        f'step {global_step}',
                        global_step=global_step
                    )
                elif train_step % FLAGS.summary_frequency == 0:
                    _, summary = sess.run([optimize, merged], feed)
                    train_writer.add_summary(summary, global_step)
                else:
                    sess.run([optimize], feed)

    finally:
        # For now, TPU sessions must be shutdown separately from
        # closing the session.
        if FLAGS.tpu_address is not None:
            sess.run(tf.contrib.tpu.shutdown_system())
        sess.close()


if __name__ == "__main__":
    define_flags()
    main()
