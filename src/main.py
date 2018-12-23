# @title The Big File { display-mode: "form" }
from __future__ import absolute_import
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


def construct_vae(data, original_dim):
    data, test = data

    make_encoder = tf.make_template('encoder', make_encoder_layer)
    make_decoder = tf.make_template('decoder', make_decoder_layer)
    make_latent_likelihood = tf.make_template(
        'latent_likelihood', make_latent_likelihood_layer
    )
    make_latent_distribution = tf.make_template(
        'distribution', make_latent_distribution_layer
    )

    distribution = make_latent_distribution()

    # Define the model.
    latent = make_encoder(data)    # (batch, z_dims, hidden_depth)
    latent = layers.Quantizer()(latent)
    x_tilde = make_decoder(latent)
    num_pixels = np.prod(original_dim[:2])

    stopped_latents = tf.stop_gradient(latent)
    likelihoods = distribution.cdf(
        tf.clip_by_value(stopped_latents + 0.5 / 255.0, 0.0, 1.0)
    ) - distribution.cdf(
        tf.clip_by_value(stopped_latents - 0.5 / 255.0, 0.0, 1.0)
    )

    log_sampling_information(latent, distribution)

    with summary.SummaryScope('losses') as scope:
        bpp_flattened = tf.layers.flatten(tf.log(likelihoods))
        train_bpp = tf.reduce_mean(tf.reduce_sum(bpp_flattened, axis=[1]))
        train_bpp /= -np.log(2) * num_pixels
        train_mse = tf.reduce_mean(tf.squared_difference(test, x_tilde))
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
    if tf.gfile.Exists(FLAGS.summaries_dir):
        files_to_remove = [
            f'{FLAGS.summaries_dir}/{dir}_{FLAGS.run_type}'
            for dir in [FLAGS.train_dir, FLAGS.test_dir]
        ]

        for file_to_remove in files_to_remove:
            if tf.gfile.Exists(file_to_remove):
                print(f'removing: {file_to_remove}')
                tf.gfile.DeleteRecursively(file_to_remove)


def print_params():
    print('---------------')
    total_count = util.count_parameters()
    print(f'number of parameters in vae: {total_count}')

    params = {
        scope: util.count_parameters(scope) for scope in ['encoder', 'decoder']
    }

    params['auxiliary nodes'] = total_count - sum(
        (count for _, count in params.items() if count is not None)
    )

    for scope, count in params.items():
        print('---------------')
        print(f'number of parameters in {scope}: {count}')

    print('---------------')


def dataset_queue():
    train_data = retrieval.cifar_input_fn()
    test_data = retrieval.cifar_input_fn(test=True)

    train_iter = train_data.make_one_shot_iterator()
    train_next = train_iter.get_next()
    test_iter = test_data.make_one_shot_iterator()
    test_next = test_iter.get_next()

    use_train_data = tf.placeholder(tf.bool)
    return (
        use_train_data,
        tf.cond(use_train_data, lambda: train_next, lambda: test_next),
    )


def main():
    print('-----FLAGS----')
    pprint.pprint(FLAGS.__dict__)
    print('--------------')

    use_train_data, data = dataset_queue()
    (x_input, elbo, code, optimize, merged, images) = construct_vae(
        data,
        original_dim=DIM,
    )

    print_params()

    if FLAGS.tpu_address is None:
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    else:
        sess = tf.Session(
            FLAGS.tpu_address, config=tf.ConfigProto(log_device_placement=True)
        )

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

        for epoch in range(FLAGS.epochs):
            start_time = time.time()

            feed = {
                use_train_data: False,
            }

            test_elbo, summary, images_summary = sess.run([
                elbo, merged, images
            ], feed)
            test_writer.add_summary(summary, FLAGS.train_steps * epoch)
            test_writer.add_summary(images_summary, FLAGS.train_steps * epoch)

            # training step
            for train_step in range(FLAGS.train_steps):
                feed = {
                    use_train_data: True,
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
