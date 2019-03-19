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
import summary
import util
import layers
import retrieval
import time

tf.reset_default_graph()

MIN_SIZE = 8


def subset(keys, dictionary):
    return {key: dictionary[key] for key in keys if key in dictionary}


def in_categories(prefixes, dictionary):
    return {
        key: dictionary[key]
        for key in dictionary.keys()
        if any(key.startswith(prefix) for prefix in prefixes)
    }


def reuse_layer(scope_name, layer_func):

    templates = [
        layer_func(),
        layer_func(),
    ]

    def res(size, input):
        if size <= MIN_SIZE:
            return templates[0](input)
        else:
            return templates[1](input)

    return res


def make_template(scope_name, layer_func):

    template = tf.make_template(scope_name, layer_func())

    def res(size, input):
        if size > MIN_SIZE:
            return template(input)
        else:
            with tf.name_scope("first_" + scope_name):
                return layer_func()(input)

    return res


def no_reuse(scope_name, layer_func):

    def res(input):
        with tf.name_scope(scope_name):
            return layer_func()(input)

    return lambda size, input: res(input)


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

    METRICS = 'metrics'
    TRAIN = 'train'
    SUMMARIES = 'summary'

    DATA = 'summary/data'

    def __init__(self, data, original_dim):
        self.data, self.test = data
        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            self.image_summary_idx = tf.random_uniform(
                [10],
                minval=0,
                maxval=tf.shape(data, out_type=tf.int32)[0],
                dtype=tf.int32,
            )
            self.build_reused_layers()
            self.new_original_dim(original_dim)

    def tensors(self):
        return {
            'metrics/loss': self.train_loss,
            'train/op': self.train_op,
            'summary/data': self.merged,
            'summary/images': self.images_summary,
            'metrics/bpp': self.train_bpp,
            'metrics/mse': self.train_mse,
            'metrics/psnr': self.train_psnr,
        }

    def new_original_dim(self, original_dim):
        self.original_dim = original_dim
        _, self.images, self.outputs = self.build(
            self.data,
            original_dim[0],
        )
        self.build_losses()

    def build(self, input, size):
        '''recursively builds residual autoencoder'''
        if size <= MIN_SIZE:    # TODO make this a flag
            images = []
            outputs = []
            residual = input
            predicted_output = tf.zeros_like(input)
            if FLAGS.debug >= 1:
                print(f'input: {residual.shape}')
        else:
            # recursive case
            pooled_input = tf.keras.layers.AvgPool2D()(input)

            if FLAGS.debug >= 1:
                print(f'downsample: {pooled_input.shape}')

            pooled_output, images, outputs = self.build(pooled_input, size // 2)

            with tf.variable_scope(f'size_{size}', reuse=tf.AUTO_REUSE):
                predicted_output = tf.image.resize_bilinear(
                    pooled_output, 2 * tf.shape(pooled_output)[1:3]
                )
            if FLAGS.debug >= 1:
                print(f'upsample: {predicted_output.shape}')
            residual = input - predicted_output

        latent = self.encoder(size, residual)
        if FLAGS.debug >= 1:
            print(f'encoded: {latent.shape}')
        latent = layers.Quantizer()(latent)

        decoded = self.decoder(size, latent)

        if size <= MIN_SIZE:    # TODO make this a flag
            output = tf.nn.relu(decoded)
        else:
            output = tf.nn.relu(decoded + predicted_output)

        images.append(
            tf.concat(
                [
                    input,
                    predicted_output,
                    residual,
                    decoded,
                    output,
                ],
                axis=2,
            )
        )

        outputs.append({
            'input': input,
            'output': output,
            'likelihood': self.likelihoods(size, latent),
        })

        if FLAGS.debug >= 1:
            print(f'decoded: {output.shape}')
        return (output, images, outputs)

    def build_reused_layers(self):
        if not FLAGS.reuse:
            template = no_reuse
        else:
            template = reuse_layer

        self.encoder = template(
                'encoder',
                lambda: layers.Encoder(FLAGS.channel_dims, FLAGS.hidden_dims,)
        )
        self.decoder = template(
            'decoder',
            lambda: layers.Decoder(FLAGS.channel_dims),
        )
        self.upsampler = template(
            'upsampler',
            lambda: layers.Upsampler(FLAGS.channel_dims),
        )

        self.likelihoods = template(
            'likelihoods',
            lambda: layers.LatentDistribution(),
        )

    def build_losses(self):
        num_pixels = np.prod(self.original_dim[:2])

        with summary.SummaryScope('losses') as scope:
            expected_bits_per_image = tf.reduce_sum(
                [
                    tf.reduce_sum(
                        tf.log(layer['likelihood'] + 1e-12),
                        axis=[1, 2, 3],
                    ) for layer in self.outputs
                ],
                axis=[0],
            )

            train_bpp = tf.reduce_mean(expected_bits_per_image)
            train_bpp /= -np.log(2) * num_pixels

            # train_mse = tf.reduce_mean(
            #     tf.losses.mean_squared_error(
            #         self.outputs[-1]['output'],
            #         self.outputs[-1]['input'],
            #         1.0,
            #     )
            # )

            train_mse = tf.reduce_sum([
                tf.reduce_mean(
                    tf.squared_difference(
                        layer['input'],
                        layer['output'],
                    )
                ) for idx, layer in enumerate(self.outputs[-1:])
            ])

            train_mse *= 255**2 / num_pixels
            train_loss = train_mse * 0.05 + train_bpp

            train_psnr = tf.reduce_mean(
                tf.image.psnr(
                    self.outputs[-1]['output'],
                    self.outputs[-1]['input'],
                    1.0,
                )
            )

            scope['bpp'] = train_bpp
            scope['mse'] = train_mse
            scope['loss'] = train_loss
            scope['psnr'] = train_psnr

        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate) \
            .minimize(train_loss)

        merged = tf.summary.merge_all()

        images_summary = tf.summary.merge([
            tf.summary.image(
                f'comparison_{idx}',
                image,
                max_outputs=FLAGS.batch_size,
            ) for idx, image in enumerate(self.images)
        ])
        self.train_loss = train_loss
        self.train_op = train_op
        self.merged = merged
        self.images_summary = images_summary
        self.train_bpp = train_bpp
        self.train_mse = train_mse
        self.train_psnr = train_psnr


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
        current_size = MIN_SIZE
    else:
        current_size = FLAGS.fixed_size

    crop_size, use_train_data, initializers, data = dataset_queue(
        crop_size=FLAGS.fixed_size
    )

    compressor = Compressor(
        data,
        original_dim=(current_size, current_size, 3),
    )
    tensors = compressor.tensors()

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
            size: 2**(i + 4)
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
                tensors = compressor.tensors()
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

            test_output = sess.run(
                in_categories([Compressor.METRICS, Compressor.SUMMARIES],
                              tensors),
                feed,
            )
            test_writer.add_summary(
                test_output[Compressor.DATA],
                FLAGS.train_steps * epoch,
            )
            test_writer.add_summary(
                test_output['summary/images'],
                FLAGS.train_steps * epoch,
            )

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
                    train_output = sess.run(
                        in_categories(
                            [
                                Compressor.METRICS,
                                Compressor.TRAIN,
                                Compressor.DATA,
                            ],
                            tensors,
                        ),
                        feed,
                        options=run_options,
                        run_metadata=run_metadata,
                    )
                    train_writer.add_summary(
                        train_output[Compressor.DATA], global_step
                    )
                    train_writer.add_run_metadata(
                        run_metadata,
                        f'step {global_step}',
                        global_step=global_step
                    )
                elif train_step % FLAGS.summary_frequency == 0:
                    train_output = sess.run(
                        in_categories(
                            [
                                Compressor.METRICS,
                                Compressor.TRAIN,
                                Compressor.DATA,
                            ],
                            tensors,
                        ),
                        feed,
                    )
                    train_writer.add_summary(
                        train_output[Compressor.DATA], global_step
                    )
                else:
                    train_output = sess.run(
                        in_categories(
                            [Compressor.METRICS, Compressor.TRAIN],
                            tensors,
                        ),
                        feed,
                    )

                if FLAGS.progress:
                    print(
                        f'Epoch: {epoch} '
                        f'step: {train_step} '
                        f'train mse: {train_output["metrics/mse"]:.5f} '
                        f'train bpp: {train_output["metrics/bpp"]:.5f} ',
                        end='\r'
                    )

            print(
                f'Epoch: {epoch} '
                f'mse: {test_output["metrics/mse"]:.6f} '
                f'bpp: {test_output["metrics/bpp"]:.6f} '
                f'psnr: {test_output["metrics/psnr"]:.6f} '
                f'time elapsed: {time.time() - start_time:.3f} seconds '
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
