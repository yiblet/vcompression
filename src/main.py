# @title The Big File { display-mode: "form" }
from __future__ import absolute_import
import os
import pprint
import tensorflow as tf
import tensorflow_probability as tfp
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


def make_latent_distribution_layer():
    tfd = tfp.distributions
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
        scale = tf.nn.softplus(
            tf.get_variable(
                name='logistic_scale_variables',
                shape=[FLAGS.categorical_dims],
            )
        )
        scope['categorical'] = categorical
        scope['loc'] = loc
        scope['scale'] = scale

        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=categorical),
            components_distribution=tfd.Normal(
                loc=loc,
                scale=scale,
            )
        )


def make_encoder_layer(data, channel, original_dim):
    data = tf.reshape(data, (-1, *original_dim))
    return layers.Encoder(channel)(data)


def make_decoder_layer(code, channel):
    return layers.Decoder(channel)(code)


def log_sampling_information(latent, distribution):
    with summary.SummaryScope('samples') as scope:
        samples = tf.layers.flatten(latent)
        scope['latent_samples'] = samples
        scope['distribution_samples'] = distribution.sample(
            samples.get_shape()[-1]
        )


def construct_vae(original_dim, channel):

    data = tf.placeholder(tf.float32, [None, *original_dim])

    make_encoder = tf.make_template('encoder', make_encoder_layer)
    make_decoder = tf.make_template('decoder', make_decoder_layer)
    make_latent_distribution = tf.make_template(
        'distribution', make_latent_distribution_layer
    )

    distribution = make_latent_distribution()

    # Define the model.
    latent = make_encoder(
        data, channel, original_dim
    )    # (batch, z_dims, hidden_depth)
    latent = layers.Quantizer()(latent)
    x_tilde = make_decoder(latent, channel)
    num_pixels = original_dim[0] * original_dim[1]

    stopped_latents = tf.stop_gradient(latent)
    likelihoods = distribution.prob(stopped_latents)
    log_sampling_information(stopped_latents, distribution)
    with summary.SummaryScope('losses') as scope:
        train_bpp = tf.reduce_mean(
            tf.reduce_sum(tf.layers.flatten(tf.log(likelihoods)), axis=[1])
        )

        train_bpp /= -np.log(2) * num_pixels
        train_mse = tf.reduce_mean(tf.squared_difference(data, x_tilde))
        train_mse *= 255**2 / num_pixels
        train_loss = train_mse * 0.1 + train_bpp
        scope['bpp'] = train_bpp
        scope['mse'] = train_mse
        scope['loss'] = train_loss

    main_step = tf.train.AdamOptimizer(FLAGS.learning_rate) \
        .minimize(train_loss)

    random_sub_batch_dims = tf.random_uniform(
        [10],
        minval=0,
        maxval=tf.shape(data, out_type=tf.int32)[0],
        dtype=tf.int32,
    )

    latent_random_sub_batch = tf.gather(latent, random_sub_batch_dims)
    data_random_sub_batch = tf.gather(data, random_sub_batch_dims)

    generated = make_decoder(latent_random_sub_batch, channel)

    merged = tf.summary.merge_all()

    image_comparison = tf.concat([data_random_sub_batch, generated], 2)
    comparison_summary = tf.summary.image(
        'comparison', image_comparison, max_outputs=10
    )

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
        original_dim=DIM,
        channel=64,
    )

    print('---------------')
    util.print_param_count()
    print('---------------')
    util.print_param_count('encoder')
    print('---------------')
    util.print_param_count('decoder')
    print('---------------')

    if FLAGS.debug:
        return

    train, test = retrieval.load_data()

    if FLAGS.tpu_address is None:
        sess = tf.Session()
    else:
        sess = tf.Session(FLAGS.tpu_address)

    if os.path.exists(FLAGS.summaries_dir):
        rm_commands = [[
            'rm', '-rf', f'{FLAGS.summaries_dir}/{dir}_{FLAGS.run_type}'
        ] for dir in [FLAGS.train_dir, FLAGS.test_dir]]

        for rm_command in rm_commands:
            print(f'running: {rm_command}')
            subprocess.Popen(
                rm_command,
                shell=True,
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE
            ).communicate()

    train_writer = tf.summary.FileWriter(
        f'{FLAGS.summaries_dir}/{FLAGS.train_dir}_{FLAGS.run_type}', sess.graph
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
                test[np.random.choice(test.shape[0], 100, replace=False), ...].
                reshape([-1, *DIM])
            }

            test_elbo, summary, images_summary = sess.run([
                elbo, merged, images
            ], feed)
            test_writer.add_summary(summary, FLAGS.train_steps * epoch)
            test_writer.add_summary(images_summary, FLAGS.train_steps * epoch)
            print(f'Epoch {epoch} elbo: {test_elbo}')

            # training step
            for train_step in range(FLAGS.train_steps):
                feed = {
                    x_input:
                    train[np.random.choice(
                        train.shape[0], FLAGS.batch_size, replace=False
                    ), ...].reshape([-1, *DIM])
                }

                global_step = FLAGS.train_steps * epoch + train_step
                if global_step == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE
                    )
                    run_metadata = tf.RunMetadata()
                    _, summary = sess.run([optimize, merged],
                                          feed,
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_summary(summary, global_step)
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
