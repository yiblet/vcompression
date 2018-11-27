# @title The Big File { display-mode: "form" }
from __future__ import absolute_import
import os
import shutil
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
import time

tf.reset_default_graph()


def make_latent_likelihood_layer(latent):
    tfd = tfp.distributions

    distribution_layer = layers.LatentDistribution()

    return (distribution_layer(latent), distribution_layer.distribution)


def make_encoder_layer(data):
    return layers.Encoder(FLAGS.channel_dims, FLAGS.hidden_dims)(data)


def make_decoder_layer(code):
    return layers.Decoder(FLAGS.channel_dims)(code)


def log_sampling_information(latent, distribution):
    with summary.SummaryScope('samples') as scope:
        samples = tf.layers.flatten(latent)
        scope['latent_samples'] = samples
        scope['distribution_samples'] = distribution.sample(
            samples.get_shape()[-1]
        )


def log_images_summary(data, latent, make_decoder):
    random_sub_batch_dims = tf.random_uniform(
        [10],
        minval=0,
        maxval=tf.shape(data, out_type=tf.int32)[0],
        dtype=tf.int32,
    )
    latent_random_sub_batch = tf.gather(latent, random_sub_batch_dims)
    data_random_sub_batch = tf.gather(data, random_sub_batch_dims)
    generated = make_decoder(latent_random_sub_batch)
    image_comparison = tf.concat([data_random_sub_batch, generated], 2)
    return tf.summary.image('comparison', image_comparison, max_outputs=10)


def construct_vae(original_dim):

    data = tf.placeholder(tf.float32, [None, *original_dim])

    make_encoder = tf.make_template('encoder', make_encoder_layer)
    make_decoder = tf.make_template('decoder', make_decoder_layer)
    make_latent_likelihood = tf.make_template(
        'distribution', make_latent_likelihood_layer
    )

    # Define the model.
    latent = make_encoder(data)    # (batch, z_dims, hidden_depth)
    latent = layers.Quantizer()(latent)
    likelihoods, distribution = make_latent_likelihood(latent)
    x_tilde = make_decoder(latent)
    num_pixels = np.prod(original_dim[:2])

    log_sampling_information(latent, distribution)

    with summary.SummaryScope('losses') as scope:
        bpp_flattened = tf.layers.flatten(tf.log(likelihoods))
        train_bpp = tf.reduce_mean(tf.reduce_sum(bpp_flattened, axis=[1]))
        train_bpp /= -np.log(2) * num_pixels
        train_mse = tf.reduce_mean(tf.squared_difference(data, x_tilde))
        train_mse *= 255**2 / num_pixels
        train_loss = train_mse * 0.05 + train_bpp
        scope['likelihoods'] = likelihoods
        scope['bpp'] = train_bpp
        scope['mse'] = train_mse
        scope['loss'] = train_loss

    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate) \
        .minimize(train_loss)

    merged = tf.summary.merge_all()

    images_summary = log_images_summary(data, latent, make_decoder)

    return (data, train_loss, latent, train_op, merged, images_summary)


def clean_log_directories():
    if os.path.exists(FLAGS.summaries_dir):
        files_to_remove = [
            f'{FLAGS.summaries_dir}/{dir}_{FLAGS.run_type}'
            for dir in [FLAGS.train_dir, FLAGS.test_dir]
        ]

        for file_to_remove in files_to_remove:
            print(f'removing: {file_to_remove}')
            shutil.rmtree(file_to_remove)


def print_params():
    print('---------------')
    total_count = util.print_param_count()
    print(f'number of parameters in vae: {total_count}')

    params = {
        scope: util.print_param_count(scope)
        for scope in ['encoder', 'decoder']
    }

    params['auxiliary nodes'] = total_count - sum(
        count for _, count in params.items()
    )

    for scope, count in params.items():
        print('---------------')
        print(f'number of parameters in {scope}: {count}')

    print('---------------')


def main():
    print('-----FLAGS----')
    pprint.pprint(FLAGS.__dict__)
    print('--------------')

    (x_input, elbo, code, optimize, merged, images) = construct_vae(
        original_dim=DIM,
    )

    print_params()

    if FLAGS.debug:
        return

    train, test = retrieval.load_data()

    if FLAGS.tpu_address is None:
        sess = tf.Session()
    else:
        sess = tf.Session(FLAGS.tpu_address)

    clean_log_directories()

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
            start_time = time.time()
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
                    train_elbo, _, summary = sess.run([elbo, optimize, merged],
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
                    train_elbo, _, summary = sess.run([elbo, optimize, merged],
                                                      feed)
                    train_writer.add_summary(summary, global_step)
                else:
                    train_elbo, _ = sess.run([elbo, optimize], feed)

                print(
                    f'Epoch: {epoch} step: {train_step} train elbo: {train_elbo:.5f}',
                    end='\r'
                )

            print(
                f'Epoch: {epoch} elbo: {test_elbo} time elapsed: {time.time() - start_time:.2f} seconds'
            )

    finally:
        # For now, TPU sessions must be shutdown separately from
        # closing the session.
        if FLAGS.tpu_address is not None:
            sess.run(tf.contrib.tpu.shutdown_system())
        sess.close()


if __name__ == "__main__":
    define_flags()
    main()
