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

tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.ERROR)


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
    return layers.Encoder(channel, 32)(data)


def make_decoder_layer(code, channel):
    return layers.Decoder(channel)(code)


def log_sampling_information(latent, distribution):
    with summary.SummaryScope('samples') as scope:
        samples = tf.layers.flatten(latent)
        scope['latent_samples'] = samples
        scope['distribution_samples'] = distribution.sample(
            samples.get_shape()[-1]
        )


def log_image_information(features, latent, channel, make_decoder):
    with tf.name_scope('sampling'):
        random_sub_batch_dims = tf.random_uniform(
            [10],
            minval=0,
            maxval=tf.shape(features, out_type=tf.int32)[0],
            dtype=tf.int32,
        )

        latent_random_sub_batch = tf.gather(latent, random_sub_batch_dims)
        data_random_sub_batch = tf.gather(features, random_sub_batch_dims)

        generated = make_decoder(latent_random_sub_batch, channel)

        image_comparison = tf.concat([data_random_sub_batch, generated], 2)
    return tf.summary.image('comparison', image_comparison, max_outputs=10)


def create_estimator_spec(**kwargs):
    if not FLAGS.use_tpu:
        return tf.estimator.EstimatorSpec(**kwargs)
    else:
        return tf.contrib.tpu.TPUEstimatorSpec(**kwargs)


def model_fn(features, labels, mode, params):
    original_dim = features.get_shape().as_list()[1:]
    channel = params['channel']
    train_summary_dir = params['train_summary_dir']
    test_summary_dir = params['test_summary_dir']

    make_encoder = tf.make_template('encoder', make_encoder_layer)
    make_decoder = tf.make_template('decoder', make_decoder_layer)
    make_latent_distribution = tf.make_template(
        'distribution', make_latent_distribution_layer
    )

    distribution = make_latent_distribution()

    # Define the model.
    latent = make_encoder(
        features, channel, original_dim
    )    # (batch, z_dims, hidden_depth)
    latent = layers.Quantizer()(latent)
    x_tilde = make_decoder(latent, channel)
    num_pixels = original_dim[0] * original_dim[1]

    stopped_latents = tf.stop_gradient(latent)
    likelihoods = distribution.cdf(
        tf.clip_by_value(stopped_latents + 0.5 / 255.0, 0.0, 1.0)
    ) - distribution.cdf(
        tf.clip_by_value(stopped_latents - 0.5 / 255.0, 0.0, 1.0)
    )

    log_sampling_information(stopped_latents, distribution)

    with summary.SummaryScope('losses') as scope:
        bpp_flattened = tf.layers.flatten(tf.log(likelihoods))
        train_bpp = tf.reduce_mean(tf.reduce_sum(bpp_flattened, axis=[1]))

        train_bpp /= -np.log(2) * num_pixels
        train_mse = tf.reduce_mean(tf.squared_difference(features, x_tilde))
        train_mse *= 255**2 / num_pixels
        train_loss = train_mse * 0.05 + train_bpp
        scope['likelihoods'] = likelihoods
        scope['bpp'] = train_bpp
        scope['mse'] = train_mse
        scope['loss'] = train_loss

    predictions = x_tilde
    loss = train_loss
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    if FLAGS.use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    train_op = optimizer.minimize(train_loss)

    if tf.estimator.ModeKeys.EVAL:
        log_image_information(features, latent, channel, make_decoder)

    return create_estimator_spec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        evaluation_hooks=[
            util.MetadataHook(save_steps=10, output_dir=test_summary_dir),
            tf.train.SummarySaverHook(
                save_steps=1,
                output_dir=test_summary_dir,
                summary_op=tf.summary.merge_all(),
            )
        ]
    )


def gen_input_fns():
    train, test = retrieval.load_data()

    return (
        tf.estimator.inputs.numpy_input_fn(
            train,
            train,
            batch_size=FLAGS.batch_size,
            shuffle=True,
        ),
        tf.estimator.inputs.numpy_input_fn(
            test,
            test,
            batch_size=FLAGS.batch_size,
            shuffle=True,
        ),
    )


def clean_logdir():
    if os.path.exists(FLAGS.summaries_dir):
        files_to_remove = [
            f'{FLAGS.summaries_dir}/{dir}_{FLAGS.run_type}'
            for dir in [FLAGS.test_dir, FLAGS.train_dir]
        ]

        for file_to_remove in files_to_remove:
            print(f'removing: {file_to_remove}')
            if os.path.isdir(file_to_remove):
                shutil.rmtree(file_to_remove)


def construct_params(channel=64):
    return {
        'channel':
        64,
        'train_summary_dir':
        f'{FLAGS.summaries_dir}/{FLAGS.train_dir}_{FLAGS.run_type}',
        'test_summary_dir':
        f'{FLAGS.summaries_dir}/{FLAGS.test_dir}_{FLAGS.run_type}',
    }


def construct_estimator(params):
    params = construct_params(channel=64)
    train_summary_dir = params['train_summary_dir']
    test_summary_dir = params['test_summary_dir']
    if not FLAGS.use_tpu:
        print('NOT using tpu')
        config = tf.estimator.RunConfig(
            model_dir=train_summary_dir,
            save_summary_steps=200,
        )
        estimator = tf.estimator.Estimator(
            model_fn, params=params, config=config
        )

    else:
        print('using tpu')
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_address
        )
        tpu_config = tf.contrib.tpu.TPUConfig()

        config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=train_summary_dir,
            save_summary_steps=200,
            tpu_config=tpu_config,
        )

        estimator = tf.contrib.tpu.TPUEstimator(
            model_fn,
            params=params,
            config=config,
            train_batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.batch_size,
        )
    return estimator


def main():
    if not FLAGS.use_tpu:
        sess = tf.Session()
    else:
        sess = tf.Session(FLAGS.tpu_address)

    clean_logdir()

    params = construct_params(channel=64)
    train_input_fn, eval_input_fn = gen_input_fns()

    train_summary_dir = params['train_summary_dir']
    test_summary_dir = params['test_summary_dir']

    estimator = construct_estimator(params)

    try:
        sess.run(tf.global_variables_initializer())

        if FLAGS.use_tpu:
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

        test_writer = tf.summary.FileWriter(test_summary_dir)

        for epoch in range(FLAGS.epochs):
            estimator.evaluate(eval_input_fn, steps=1)
            estimator.train(train_input_fn, steps=FLAGS.train_steps)
            print(f'epoch: {epoch}')

    finally:
        # For now, TPU sessions must be shutdown separately from
        # closing the session.
        if FLAGS.tpu_address is not None:
            sess.run(tf.contrib.tpu.shutdown_system())
        sess.close()


if __name__ == "__main__":
    define_flags()
    main()
