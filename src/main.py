# @title The Big File { display-mode: "form" }
from __future__ import absolute_import
import os
import pprint
import tensorflow as tf
import numpy as np
import subprocess
import sys

from global_variables import *
import entropy
import summary
import util
import layers
import retrieval

tf.reset_default_graph()


def construct_vae(original_dim, hidden_depth=3):
    import tensorflow_probability as tfp
    tfd = tfp.distributions

    def make_encoder(data):
        with summary.SummaryScope('Q-probability') as scope:
            x = tf.reshape(data, (-1, *original_dim))
            scope['input'] = x
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
                        activation=tf.nn.sigmoid,
                    ),
                ],
            )

        return x

    def make_latent_distribution():
        with summary.SummaryScope('latent_distributions') as scope:
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
                components_distribution=tfd.Normal(
                    loc=loc,
                    scale=scale,
                )
            )

    def make_decoder(code):
        with summary.SummaryScope('P-probability') as scope:
            scope['input'] = code
            logit = scope.sequential(
                code,
                [
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

    @tf.custom_gradient
    def quantize(latent):
        expand = latent * 128.0
        expand = tf.clip_by_value(expand, -128, 128)
        expand = tf.round(expand)
        expand /= 128.0

        def grad(dy):
            return dy * (1 - tf.cos(128.0 * np.pi * latent))

        return expand, grad

    data = tf.placeholder(tf.float32, [None, *original_dim])

    make_encoder = tf.make_template('encoder', make_encoder)
    make_decoder = tf.make_template('decoder', make_decoder)
    make_latent_distribution = tf.make_template(
        'distribution', make_latent_distribution
    )

    distribution = make_latent_distribution()

    # Define the model.
    latent = make_encoder(data)  # (batch, z_dims, hidden_depth)
    latent = quantize(latent)

    stopped_latents = tf.stop_gradient(latent)
    likelihoods = distribution.prob(stopped_latents)

    with summary.SummaryScope('samples') as scope:
        samples = tf.layers.flatten(stopped_latents)
        scope['latent_samples'] = samples
        scope['distribution_samples'] = distribution.sample(
            samples.get_shape()[-1]
        )

    # entropy_bottleneck = entropy.entropy_models.EntropyBottleneck()
    # latent_tilde, likelihoods = entropy_bottleneck(latent, training=True)

    x_tilde = make_decoder(latent)

    num_pixels = original_dim[0] * original_dim[1]

    # Define the loss.
    with summary.SummaryScope('losses') as scope:
        train_bpp = tf.reduce_mean(tf.reduce_sum(
            tf.log(likelihoods), axis=[1, 2])
        )
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
        subprocess.Popen(
            f'rm -r {FLAGS.summaries_dir}/{FLAGS.test_dir}_{FLAGS.run_type}',
            shell=True,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
        ).communicate()
        subprocess.Popen(
            f'rm -r {FLAGS.summaries_dir}/{FLAGS.train_dir}_{FLAGS.run_type}',
            shell=True,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
        ).communicate()

    train_writer = tf.summary.FileWriter(
        f'{FLAGS.summaries_dir}/{FLAGS.train_dir}_{FLAGS.run_type}',
        sess.graph
    )
    test_writer = tf.summary.FileWriter(
        f'{FLAGS.summaries_dir}/{FLAGS.test_dir}_{FLAGS.run_type}',
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
