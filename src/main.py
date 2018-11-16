# @title The Big File { display-mode: "form" }
from __future__ import absolute_import
import os
import pprint
import tensorflow as tf
import numpy as np
import subprocess
import sys
import entropy

from global_variables import *
import util
import layers
import retrieval

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
                    ),
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
                    tf.keras.layers.Activation('relu', name='relu_4'),
                    tf.layers.Conv2D(
                        128,
                        [2, 2],
                        [1, 1],
                        name='conv_5',
                        activation=tf.nn.sigmoid,
                    ),
                ],
            )

            dims = np.prod(x.get_shape().as_list()[1:])
            assert (dims == FLAGS.z_dims)
            latent = tf.reshape(x, (-1, dims))

        return latent

    def make_latent_distribution():
        with SummaryScope('latent_distributions') as scope:
            categorical = tf.get_variable(
                name='categorical_distribution',
                shape=[FLAGS.categorical_dims],
            )
            categorical = tf.nn.softmax(categorical)
            loc = tf.get_variable(
                name='logistic_loc_variables',
                shape=[FLAGS.categorical_dims],
            )
            scale = tf.nn.softplus(tf.get_variable(
                name='logistic_scale_variables',
                shape=[FLAGS.categorical_dims],
            ))
            scope['categorical'] = categorical
            scope['loc'] = loc
            scope['scale'] = scale
            return tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(
                    probs=categorical
                ),
                components_distribution=tfd.Logistic(
                    loc=loc,
                    scale=scale,
                ),
            )

    def make_decoder(code):
        with SummaryScope('P-probability') as scope:
            logit = scope.sequential(
                tf.reshape(code, (-1, 1, 1, FLAGS.z_dims)),
                [
                    tf.layers.Conv2DTranspose(
                        128,
                        [2, 2],
                        [1, 1],
                        name='deconv_1',
                        activation=None,
                    ),
                    tf.keras.layers.Activation('relu', name='relu_1'),
                    tf.layers.Conv2DTranspose(
                        128,
                        [2, 2],
                        [2, 2],
                        name='deconv_2',
                        activation=None,
                    ),
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

        return tf.nn.relu(logit)

    data = tf.placeholder(tf.float32, [None, *original_dim])

    make_encoder = tf.make_template('encoder', make_encoder)
    make_decoder = tf.make_template('decoder', make_decoder)
    make_latent_distribution = tf.make_template(
        'distribution', make_latent_distribution
    )

    distribution = make_latent_distribution()

    # Define the model.
    latent = make_encoder(data)

    stopped_latents = tf.stop_gradient(latent)
    likelihoods = distribution.cdf(stopped_latents + 0.5 / 255) - \
        distribution.cdf(stopped_latents - 0.5 / 255)

    # entropy_bottleneck = entropy.entropy_models.EntropyBottleneck()
    # latent_tilde, likelihoods = entropy_bottleneck(latent, training=True)

    x_tilde = make_decoder(latent)

    num_pixels = original_dim[0] * original_dim[1]

    # Define the loss.
    with SummaryScope('losses') as scope:
        train_bpp = tf.reduce_mean(tf.log(likelihoods))
        train_bpp /= -np.log(2) * num_pixels
        train_mse = tf.reduce_mean(tf.squared_difference(data, x_tilde))
        train_mse *= 255 ** 2 / num_pixels
        train_loss = train_mse * 0.1 + train_bpp
        scope['bpp'] = train_bpp
        scope['mse'] = train_mse
        scope['loss'] = train_loss

    main_step = tf.train.AdamOptimizer(
        FLAGS.learning_rate).minimize(train_loss)

    random_sub_batch_dims = tf.random_uniform(
        [10],
        minval=0,
        maxval=tf.shape(data,  out_type=tf.int32)[0],
        dtype=tf.int32,
    )

    latent_random_sub_batch = tf.gather(latent, random_sub_batch_dims)
    data_random_sub_batch = tf.gather(data, random_sub_batch_dims)

    generated = make_decoder(latent_random_sub_batch)

    merged = tf.summary.merge_all()

    image_comparison = tf.concat([data_random_sub_batch, generated], 2)
    comparison_summary = tf.summary.image(
        'comparison',
        image_comparison,
        max_outputs=10
    )

    # aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    # aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])
    train_op = tf.group(main_step)

    images_summary = tf.summary.merge([comparison_summary])

    return (data, train_loss, latent, train_op, merged, images_summary)


def gdn(input):
    return tf.contrib.layers.gdn(input)


def inverse_gdn(input):
    return tf.contrib.layers.gdn(input, inverse=True)


def main():
    print('-----FLAGS----')
    pprint.pprint(FLAGS.__dict__)
    print('--------------')

    (x_input, elbo, code, optimize, merged, images) = construct_vae(
        DIM
    )

    print('---------------')
    util.print_param_count()
    print('---------------')
    util.print_param_count('encoder')
    print('---------------')
    util.print_param_count('decoder')
    print('---------------')

    if FLAGS.debug:
        sys.exit()

    train, test = retrieval.load_data()

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
    print(f"logging at {FLAGS.summaries_dir}")

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
                .reshape([-1, *DIM])
            }

            test_elbo, summary, images_summary = sess.run(
                [elbo, merged, images],
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
                    .reshape([-1, *DIM])
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
