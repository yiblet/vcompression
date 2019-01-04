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


def log_sampling_information(latent, distribution):
    with summary.SummaryScope('samples') as scope:
        samples = tf.layers.flatten(latent)
        scope['latent_samples'] = samples
        scope['distribution_samples'] = distribution.sample(
            samples.get_shape()[-1]
        )


class Compressor:

    def __init__(self, data, original_dim):
        self.data, self.test = data
        self.original_dim = original_dim
        self.image_summary_idx = tf.random_uniform(
            [10],
            minval=0,
            maxval=tf.shape(data, out_type=tf.int32)[0],
            dtype=tf.int32,
        )
        self.build_reused_layers()
        self.output, self.latents, self.images = self.build(self.data)
        self.build_losses()

    def build(self, input):
        '''recursively builds residual autoencoder'''
        shape = input.shape.as_list()[1:]
        if shape[0] <= 16:    # TODO make this a flag
            latents = []
            images = []
            residual = input
            predicted_output = tf.zeros_like(input)
            if FLAGS.debug:
                print(f'input: {residual.shape}')
        else:
            # recursive case
            pooled_input = tf.keras.layers.AvgPool2D()(input)

            if FLAGS.debug:
                print(f'downsample: {pooled_input.shape}')

            pooled_output, latents, images = self.build(pooled_input)
            predicted_output = self.upsampler(pooled_output)
            if FLAGS.debug:
                print(f'upsample: {predicted_output.shape}')
            residual = input - predicted_output

        latent = self.encoder(residual)
        if FLAGS.debug:
            print(f'encoded: {latent.shape}')
        latent = layers.Quantizer()(latent)
        latents.append(latent)

        decoded = self.decoder(latent)

        if shape[0] <= 16:    # TODO make this a flag
            output = decoded
        else:
            output = decoded + predicted_output

        images.append(
            tf.concat(
                [
                    tf.gather(input, self.image_summary_idx),
                    tf.gather(predicted_output, self.image_summary_idx),
                    tf.gather(residual, self.image_summary_idx),
                    tf.gather(decoded, self.image_summary_idx),
                    tf.gather(output, self.image_summary_idx),
                ],
                axis=2,
            )
        )

        if FLAGS.debug:
            print(f'decoded: {output.shape}')
        return (output, latents, images)

    def build_reused_layers(self):
        self.encoder = layers.Encoder(FLAGS.channel_dims, FLAGS.hidden_dims)
        self.decoder = layers.Decoder(FLAGS.channel_dims)
        self.likelihoods = layers.LatentDistribution()
        self.distribution = self.likelihoods.distribution
        self.upsampler = tf.layers.Conv2DTranspose(
            3,
            [2, 2],
            [2, 2],
            name='upsampler',
            activation='relu',
        )

    def build_losses(self):
        num_pixels = np.prod(self.original_dim[:2])

        with summary.SummaryScope('losses') as scope:

            expected_bits_per_image = tf.reduce_sum(
                [
                    tf.reduce_sum(
                        tf.log(self.likelihoods(latent)), axis=[1, 2, 3]
                    ) for latent in self.latents
                ],
                axis=[0],
            )
            train_bpp = tf.reduce_mean(expected_bits_per_image)
            train_bpp /= -np.log(2) * num_pixels
            train_mse = tf.reduce_mean(
                tf.squared_difference(self.test, self.output)
            )
            train_mse *= 255**2 / num_pixels
            train_loss = train_mse * 0.05 + train_bpp
            scope['bpp'] = train_bpp
            scope['mse'] = train_mse
            scope['loss'] = train_loss

        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate) \
            .minimize(train_loss)

        merged = tf.summary.merge_all()

        images_summary = tf.summary.merge([
            tf.summary.image(f'comparison_{idx}', image, max_outputs=10)
            for idx, image in enumerate(self.images)
        ])

        self.train_loss = train_loss
        self.train_op = train_op
        self.merged = merged
        self.images_summary = images_summary

    def tensors(self):
        return (
            self.train_loss,
            self.train_op,
            self.merged,
            self.images_summary,
        )


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


def dataset_queue(input_fn=retrieval.large_image_input_fn):

    train_data = input_fn()
    test_data = input_fn(test=True)

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
    elbo, optimize, merged, images = Compressor(
        data,
        original_dim=DIM,
    ).tensors()

    print_params()

    if FLAGS.tpu_address is None:
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    else:
        sess = tf.Session(
            FLAGS.tpu_address, config=tf.ConfigProto(log_device_placement=True)
        )

    clean_log_directories()

    train_writer = tf.summary.FileWriter(
        f'{FLAGS.summaries_dir}/{FLAGS.train_dir}_{FLAGS.run_type}',
        sess.graph,
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
