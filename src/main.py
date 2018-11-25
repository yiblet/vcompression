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
tf.logging.set_verbosity(tf.logging.ERROR)


def make_latent_distribution_layer(latents):
    return layers.LatentDistribution()(latents)


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


def log_image_information(features, latent, decoder):
    with tf.name_scope('sampling'):
        random_sub_batch_dims = tf.random_uniform(
            [10],
            minval=0,
            maxval=tf.shape(features, out_type=tf.int32)[0],
            dtype=tf.int32,
        )

        latent_random_sub_batch = tf.gather(latent, random_sub_batch_dims)
        data_random_sub_batch = tf.gather(features, random_sub_batch_dims)

        generated = decoder(latent_random_sub_batch)

        image_comparison = tf.concat([data_random_sub_batch, generated], 2)
    return tf.summary.image('comparison', image_comparison, max_outputs=10)


def compressor_loss(input):
    features, bpp, predictions = input

    with summary.SummaryScope('loss') as scope:
        original_dim = features.get_shape().as_list()[1:]
        num_pixels = original_dim[0] * original_dim[1]
        print(num_pixels)
        bpp_flattened = tf.layers.flatten(bpp)
        train_bpp = tf.reduce_mean(tf.reduce_sum(bpp_flattened, axis=[1]))
        train_bpp /= -np.log(2) * num_pixels

        train_mse = tf.reduce_mean(tf.squared_difference(features, predictions))
        train_mse *= 255**2 / num_pixels
        scope['mse'] = train_mse
        scope['bpp'] = train_bpp

    return train_mse * 0.05 + train_bpp


def construct_model(features):
    original_dim = features.get_shape().as_list()[1:]
    channel = FLAGS.channel_dims
    hidden_latent_dims = FLAGS.hidden_dims

    encoder = tf.make_template('encoder', make_encoder_layer)
    decoder = tf.make_template('decoder', make_decoder_layer)
    latent_distribution = tf.make_template(
        'distribution', make_latent_distribution_layer
    )

    latent = encoder(features)
    latent = layers.Quantizer()(latent)

    x_tilde = decoder(latent)
    latent_dist = layers.LatentDistribution()
    bpp = latent_dist(latent)
    log_image_information(features, latent, decoder)
    log_sampling_information(latent, latent_dist.distribution)
    loss = tf.keras.layers.Lambda(compressor_loss)([features, bpp, x_tilde])

    # model = tf.keras.Model(
    #     inputs=[features],
    #     outputs=[loss],
    # )

    # model.compile(
    #     optimizer='adam',
    #     loss=lambda _, x: x,
    # )

    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    merged = tf.summary.merge_all()
    return features, loss, train_op, merged


def input_fn(steps=None, test=False):
    """Read CIFAR input data from a TFRecord dataset."""
    batch_size = FLAGS.batch_size

    def parser(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                "image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64),
            }
        )
        image = tf.decode_raw(features["image"], tf.uint8)
        image.set_shape([3 * 32 * 32])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image = tf.transpose(tf.reshape(image, [3, 32, 32]))
        image = tf.image.rot90(image, 3)
        return image

    if test:
        location = FLAGS.tf_records_dir + '/eval.tfrecords'
    else:
        location = FLAGS.tf_records_dir + '/train.tfrecords'

    dataset = tf.data.TFRecordDataset([location])
    dataset = dataset.map(
        parser, num_parallel_calls=batch_size
    ).cache().repeat()
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(100)
    return dataset


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


class LoggerCallback(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.logs = {}
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(
            f'{FLAGS.summaries_dir}/{FLAGS.test_dir}_{FLAGS.run_type}'
        )
        self._epoch = 0

    def _fetch_callback(self, summary):
        self.writer.add_summary(summary, self._epoch)

    def on_epoch_begin(self, epoch, logs={}):

        if self.merged not in self.model.test_function.fetches:
            self.model.test_function.fetches.append(self.merged)
            self.model.test_function.fetch_callbacks[self.merged
                                                    ] = self._fetch_callback

    def on_epoch_end(self, epoch, logs={}):
        if self.merged in self.model.test_function.fetches:
            self.model.test_function.fetches.remove(self.merged)

        if self.merged in self.model.test_function.fetch_callbacks:
            self.model.test_function.fetch_callbacks.pop(self.merged)

        self._epoch += 1


def main():
    if not FLAGS.use_tpu:
        sess = tf.Session()
    else:
        sess = tf.Session(FLAGS.tpu_address)

    clean_logdir()

    train_data_iter = input_fn().make_one_shot_iterator()
    test_data_iter = input_fn(test=True).make_one_shot_iterator()

    input, loss, train_op, merged = construct_model(train_data_iter.get_next())

    train_writer = tf.summary.FileWriter(
        f'{FLAGS.summaries_dir}/{FLAGS.train_dir}_{FLAGS.run_type}', sess.graph
    )
    test_writer = tf.summary.FileWriter(
        f'{FLAGS.summaries_dir}/{FLAGS.test_dir}_{FLAGS.run_type}',
    )

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

        for epoch in range(FLAGS.epochs):
            feed = {}

            test_loss, summary = sess.run([loss, merged], feed)
            test_writer.add_summary(summary, FLAGS.train_steps * epoch)
            print(f'Epoch {epoch} elbo: {test_loss}')

            # training step
            for train_step in range(FLAGS.train_steps):
                feed = {}

                global_step = FLAGS.train_steps * epoch + train_step
                if global_step == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE
                    )
                    run_metadata = tf.RunMetadata()
                    _, summary = sess.run([train_op, merged],
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
                    _, summary = sess.run([train_op, merged], feed)
                    train_writer.add_summary(summary, global_step)
                else:
                    sess.run([train_op], feed)

        # print(model.summary())
        # model.fit(
        #     input_fn(),
        #     validation_data=input_fn(test=True),
        #     validation_steps=1,
        #     steps_per_epoch=600,
        #     epochs=1000,
        #     verbose=1,
        #     callbacks=[LoggerCallback()]
        # )

        # for epoch in range(FLAGS.epochs):
        # estimator.evaluate(
        #     lambda: input_fn(steps=1, test=True),
        #     steps=1,
        # )

        # prev = time.time()
        # estimator.train(lambda: input_fn(steps=FLAGS.train_steps), steps=1)
        # print(f'{time.time() - prev} has passed on {600} steps')

    finally:
        # For now, TPU sessions must be shutdown separately from
        # closing the session.
        if FLAGS.tpu_address is not None:
            sess.run(tf.contrib.tpu.shutdown_system())
        sess.close()


if __name__ == "__main__":
    define_flags()
    main()
