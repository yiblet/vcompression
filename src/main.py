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


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([
        tf.is_variable_initialized(var) for var in global_vars
    ])
    not_initialized_vars = [
        v for (v, f) in zip(global_vars, is_not_initialized) if not f
    ]

    if len(not_initialized_vars):
        if FLAGS.debug >= 1:
            print(f'initializing: {[(i.name) for i in not_initialized_vars]}')
        sess.run(tf.variables_initializer(not_initialized_vars))


class Compressor:

    def __init__(self, data, original_dim):
        self.data, self.test = data
        self.image_summary_idx = tf.random_uniform(
            [10],
            minval=0,
            maxval=tf.shape(data, out_type=tf.int32)[0],
            dtype=tf.int32,
        )
        self.build_reused_layers()
        self.new_original_dim(original_dim)

    def new_original_dim(self, original_dim):
        self.original_dim = original_dim
        self.output, self.latents, self.images = self.build(
            self.data, original_dim[0]
        )
        self.build_losses()

    def build(self, input, size):
        '''recursively builds residual autoencoder'''
        if size <= 16:    # TODO make this a flag
            latents = []
            images = []
            residual = input
            predicted_output = tf.zeros_like(input)
            if FLAGS.debug >= 1:
                print(f'input: {residual.shape}')
        else:
            # recursive case
            pooled_input = tf.keras.layers.AvgPool2D()(input)

            if FLAGS.debug >= 1:
                print(f'downsample: {pooled_input.shape}')

            pooled_output, latents, images = self.build(pooled_input, size // 2)
            predicted_output = tf.image.resize_bilinear(
                pooled_output, [size, size]
            )
            if FLAGS.debug >= 1:
                print(f'upsample: {predicted_output.shape}')
            residual = input - predicted_output

        latent = self.encoder(residual)
        if FLAGS.debug >= 1:
            print(f'encoded: {latent.shape}')
        latent = layers.Quantizer()(latent)
        latents.append(latent)

        decoded = self.decoder(latent)

        if size <= 16:    # TODO make this a flag
            output = tf.nn.relu(decoded)
        else:
            output = tf.nn.relu(decoded + predicted_output)

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

        if FLAGS.debug >= 1:
            print(f'decoded: {output.shape}')
        return (output, latents, images)


    def build_reused_layers(self):
        self.encoder = lambda x:  
            layers.Encoder(
                FLAGS.channel_dims,
                FLAGS.hidden_dims,
            )(3)
        # tf.make_template(
        #     'encoder', 
        #     layers.Encoder(
        #         FLAGS.channel_dims,
        #         FLAGS.hidden_dims,
        #     )
        # )
        self.decoder = tf.make_template(
            'decoder', layers.Decoder(FLAGS.channel_dims)
        )
        self.upsampler = tf.make_template(
            'upsampler', layers.Upsampler(FLAGS.channel_dims)
        )

        likelihoods = layers.LatentDistribution()
        self.likelihoods = tf.make_template('likelihoods', likelihoods)
        self.distribution = likelihoods.distribution

    def build_losses(self):
        num_pixels = np.prod(self.original_dim[:2])

        with summary.SummaryScope('losses') as scope:

            expected_bits_per_image = tf.reduce_sum(
                [
                    tf.reduce_sum(
                        tf.log(self.likelihoods(latent) + 1e-12),
                        axis=[1, 2, 3]
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
        scope: util.count_parameters(scope)
        for scope in ['encoder', 'decoder', 'upsampler']
    }

    params['auxiliary nodes'] = total_count - sum(
        (count for _, count in params.items() if count is not None)
    )

    for scope, count in params.items():
        print('---------------')
        print(f'number of parameters in {scope}: {count}')

    print('---------------')


def dataset_queue(input_fn=retrieval.large_image_input_fn, crop_size=None):
    if crop_size is None:
        crop_size = tf.placeholder(tf.int32, name='crop_size')

    train_data = input_fn(crop_size=crop_size)
    test_data = input_fn(test=True, crop_size=crop_size)

    train_iter = train_data.make_initializable_iterator()
    train_next = train_iter.get_next()
    test_iter = test_data.make_initializable_iterator()
    test_next = test_iter.get_next()

    use_train_data = tf.placeholder(tf.bool)
    return (
        crop_size,
        use_train_data,
        [train_iter.initializer, test_iter.initializer],
        tf.cond(use_train_data, lambda: train_next, lambda: test_next),
    )


def main():
    print('-----FLAGS----')
    pprint.pprint(FLAGS.__dict__)
    print('--------------')

    if FLAGS.fixed_size is None:
        current_size = 16
    else:
        current_size = FLAGS.fixed_size

    crop_size, use_train_data, initializers, data = dataset_queue(
        crop_size=FLAGS.fixed_size
    )

    compressor = Compressor(
        data,
        original_dim=(current_size, current_size, 3),
    )
    elbo, optimize, merged, images = compressor.tensors()

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
        print('globally initializing')
        if isinstance(crop_size, tf.Tensor):
            feed = {crop_size: current_size}
        else:
            feed = {}
        sess.run([tf.global_variables_initializer(), *initializers], feed)

        if FLAGS.tpu_address is not None:
            print('Initializing TPUs...')
            sess.run(tf.contrib.tpu.initialize_system())

        print('Running ops')

        resize_times = {
            size: 2**(i + 5)
            for i, size in enumerate(FLAGS.increment_size_intervals)
        }

        pprint.pprint(resize_times)

        for epoch in range(FLAGS.epochs):
            start_time = time.time()
            if epoch in resize_times and isinstance(crop_size, tf.Tensor):
                current_size = resize_times[epoch]
                if FLAGS.debug >= 1:
                    print(f'incrementing to: {current_size}')
                compressor.new_original_dim((
                    current_size,
                    current_size,
                    3,
                ))
                elbo, optimize, merged, images = compressor.tensors()
                initialize_uninitialized(sess)
                sess.run(
                    initializers,
                    {crop_size: current_size},
                )

            if isinstance(crop_size, tf.Tensor):
                feed = {
                    use_train_data: False,
                    crop_size: current_size,
                }
            else:
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
                if isinstance(crop_size, tf.Tensor):
                    feed = {
                        use_train_data: True,
                        crop_size: current_size,
                    }
                else:
                    feed = {
                        use_train_data: True,
                    }

                global_step = FLAGS.train_steps * epoch + train_step
                if global_step == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE
                    )
                    run_metadata = tf.RunMetadata()
                    train_elbo, _, summary = sess.run(
                        [elbo, optimize, merged],
                        feed,
                        options=run_options,
                        run_metadata=run_metadata,
                    )
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

                if FLAGS.progress:
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
