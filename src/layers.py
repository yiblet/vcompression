from __future__ import absolute_import
from global_variables import *
import summary
import tensorflow as tf


class Quantizer(tf.keras.layers.Layer):

    def call(self, input):

        @tf.custom_gradient
        def quantizer(latent):
            expand = latent * 255.0
            expand = tf.clip_by_value(expand, 0, 255)
            expand = tf.round(expand)
            expand /= 255.0

            def grad(dy):
                return dy * (1 - tf.cos(255.0 * np.pi * latent))

            return expand, grad

        return quantizer(input)


class ResidualBlock(tf.keras.layers.Layer):
    """Basic Residual Block"""

    def __init__(
        self,
        channels,
        kernel=[3, 3],
        activation=None,
        use_batch_norm=False,
        batch_norm_last=True,
        **kwargs
    ):
        tf.keras.layers.Layer.__init__(self, **kwargs)
        self.batch_norm_last = batch_norm_last
        self.channels = channels
        self.kernel = kernel
        self.use_batch_norm = use_batch_norm

        if not isinstance(activation, list):
            activation = [activation, activation]
        self.activation = activation

    def build(self, input_shape):

        if FLAGS.disable_residual_block:
            return

        if len(input_shape) != 4:
            raise ValueError("input needs to have rank 4")

        create_batch_norm = None
        if self.use_batch_norm:
            create_batch_norm = tf.layers.BatchNormalization

        last_batch_norm = None
        if self.use_batch_norm and (not self.batch_norm_last):
            last_batch_norm = tf.layers.BatchNormalization()

        with tf.name_scope(self.name):
            self.layers = [
                tf.layers.Conv2D(
                    self.channels,
                    self.kernel,
                    [1, 1],
                    name="conv2d_1",
                    activation=None,
                    padding="same",
                ),
                create_batch_norm,
                self.activation[0],
                tf.layers.Conv2D(
                    self.channels,
                    self.kernel,
                    [1, 1],
                    name="conv2d_2",
                    activation=None,
                    padding="same",
                ),
                last_batch_norm,
            ]

        super().build(input_shape)

    def call(self, input):
        if FLAGS.disable_residual_block:
            return input

        cur = input
        for layer in self.layers:
            if layer is not None:
                cur = layer(cur)

        return self.activation[1](cur + input)


class SummaryLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        silent=False,
        **kwargs,
    ):
        self.silent = silent
        super().__init__(**kwargs)

    def call(self, input):
        with summary.SummaryScope(self.name) as scope:
            scope['input'] = input
            output = scope.sequential(input, self.layers)
        return output


class Encoder(SummaryLayer):

    def __init__(
        self,
        channels,
        **kwargs,
    ):
        self.channels = channels
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.layers = [
            tf.layers.Conv2D(
                self.channels, [2, 2], [2, 2], name='conv_1', activation=None
            ),
            tf.keras.layers.Activation('relu', name='relu_1'),
            ResidualBlock(self.channels, kernel=[3, 3], activation=tf.nn.relu),
            tf.layers.Conv2D(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_2',
                activation=None,
            ),
            tf.keras.layers.Activation('relu', name='relu_2'),
            ResidualBlock(self.channels, kernel=[3, 3], activation=tf.nn.relu),
            tf.layers.Conv2D(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_3',
                activation=None,
            ),
            tf.keras.layers.Activation('relu', name='relu_3'),
            ResidualBlock(self.channels, kernel=[3, 3], activation=tf.nn.relu),
            tf.layers.Conv2D(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_4',
                activation=tf.nn.sigmoid,
            ),
        ]
        super().build(input_shape)


class Decoder(SummaryLayer):

    def __init__(
        self,
        channels,
        **kwargs,
    ):
        self.channels = channels
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.layers = [
            tf.layers.Conv2DTranspose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_2',
                activation=None,
            ),
            tf.keras.layers.Activation('relu', name='relu_2'),
            ResidualBlock(self.channels, kernel=[3, 3], activation=tf.nn.relu),
            tf.layers.Conv2DTranspose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_3',
                activation=None,
            ),
            tf.keras.layers.Activation('relu', name='relu_3'),
            ResidualBlock(self.channels, kernel=[3, 3], activation=tf.nn.relu),
            tf.layers.Conv2DTranspose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_4',
                activation=None,
            ),
            tf.keras.layers.Activation('relu', name='relu_4'),
            ResidualBlock(self.channels, kernel=[3, 3], activation=tf.nn.relu),
            tf.layers.Conv2DTranspose(
                self.channels,
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
            tf.keras.layers.Activation('relu', name='relu_6')
        ]
        super().build(input_shape)
