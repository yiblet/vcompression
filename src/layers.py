from __future__ import absolute_import
from global_variables import *
import summary
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from spectral_norm import ConvSN2D, ConvSN2DTranspose


def conv2d(*args, **kwargs):
    if FLAGS.enable_spectral_normalization:
        return ConvSN2D(*args, **kwargs)
    else:
        return tf.layers.Conv2D(*args, **kwargs)


def conv2d_transpose(*args, **kwargs):
    if FLAGS.enable_spectral_normalization:
        return ConvSN2DTranspose(*args, **kwargs)
    else:
        return tf.layers.Conv2DTranspose(*args, **kwargs)


@tf.custom_gradient
def scale_gradient(input):

    def grad(dy):
        return dy * FLAGS.latent_gamma

    return input, grad


class SpectralNorm(tf.keras.constraints.Constraint):

    def __init__(self, iteration=1):
        self.iteration = iteration
        self.u = tf.get_variable(
            "u", [1, w_shape[-1]],
            initializer=tf.random_normal_initializer(),
            trainable=False
        )

    def __call__(self, w):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u_hat = self.u
        v_hat = None
        for i in range(self.iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([self.u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm

    def get_config(self):
        return {
            "iteration": self.iteration,
            "u": self.u,
        }


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


class LatentDistribution(tf.keras.layers.Layer):

    @property
    def distribution(self):
        if not self.built:
            self.build(None)

        return self._distribution

    def build(self, input_shape):
        tfd = tfp.distributions

        with summary.SummaryScope('latent_distributions') as scope:
            categorical = self.add_weight(
                name='categorical_distribution',
                shape=[FLAGS.categorical_dims],
                trainable=True,
            )
            categorical = tf.nn.softmax(categorical)
            loc = self.add_weight(
                name='logistic_loc_variables',
                shape=[FLAGS.categorical_dims],
                trainable=True,
            )
            scale = tf.nn.softplus(
                self.add_weight(
                    name='logistic_scale_variables',
                    shape=[FLAGS.categorical_dims],
                    trainable=True,
                )
            )
            scope['categorical'] = categorical
            scope['loc'] = loc
            scope['scale'] = scale

        self.vars = [categorical, loc, scale]
        self._distribution = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=categorical),
            components_distribution=tfd.Normal(
                loc=loc,
                scale=scale,
            )
        )

        self.built = True

    def call(self, latent):

        stopped_latents = scale_gradient(latent)
        likelihoods = self._distribution.cdf(
            tf.clip_by_value(stopped_latents + 0.5 / 255.0, 0.0, 1.0)
        ) - self._distribution.cdf(
            tf.clip_by_value(stopped_latents - 0.5 / 255.0, 0.0, 1.0)
        )

        return likelihoods


class ResidualBlock(tf.keras.Model):
    """Basic Residual Block"""

    def __init__(
        self,
        channels,
        kernel=[3, 3],
        activation=None,
        use_batch_norm=False,
        batch_norm_last=True,
        trainable=True,
        **kwargs
    ):
        super().__init__(self, **kwargs)
        self.trainable = True
        self.batch_norm_last = batch_norm_last
        self.channels = channels
        self.kernel = kernel
        self.use_batch_norm = use_batch_norm

        if not isinstance(activation, list):
            activation = [activation, activation]
        self.activation = activation
        self.build(None)

    def build(self, _):
        if FLAGS.disable_residual_block:
            return

        create_batch_norm = None
        if self.use_batch_norm:
            create_batch_norm = tf.layers.BatchNormalization

        last_batch_norm = None
        if self.use_batch_norm and (not self.batch_norm_last):
            last_batch_norm = tf.layers.BatchNormalization()

        with tf.name_scope(self.name):
            self.model_layers = [
                conv2d(
                    self.channels,
                    self.kernel,
                    dilation_rate=[2, 2],
                    name="conv2d_1",
                    activation=None,
                    padding="same",
                    trainable=self.trainable,
                ),
                create_batch_norm,
                self.activation[0],
                conv2d(
                    self.channels,
                    self.kernel,
                    name="conv2d_2",
                    activation=None,
                    padding="same",
                    trainable=self.trainable,
                ),
                last_batch_norm,
            ]

    def call(self, input):
        if FLAGS.disable_residual_block:
            return input

        cur = input
        for layer in self.model_layers:
            if layer is not None:
                cur = layer(cur)

        return self.activation[1](cur + input)


class SummaryModel(tf.keras.Model):

    def __init__(
        self,
        silent=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.silent = silent

    def build(self, _):
        pass

    def call(self, input):
        with summary.SummaryScope(self.name) as scope:
            scope['input'] = input
            output = scope.sequential(input, self.model_layers)
        return output


class Encoder(SummaryModel):

    def __init__(
        self,
        channels,
        hidden,
        activation='relu',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.hidden = hidden
        if activation == 'relu':
            self.activation = tf.keras.layers.Activation('relu')
        elif activation == 'tanh':
            self.activation = tf.keras.layers.Activation('tanh')
        else:
            raise ValueError('unsupported activation')
        self.build(None)

    def build(self, input_shape):
        self.model_layers = [
            conv2d(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_1',
                activation=None,
            ),
            self.activation,
            ResidualBlock(
                self.channels,
                kernel=[2, 2],
                activation=self.activation,
                use_batch_norm=False,
            ),
            conv2d(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_2',
                activation=None,
            ),
            self.activation,
            ResidualBlock(
                self.channels,
                kernel=[2, 2],
                activation=self.activation,
                use_batch_norm=False,
            ),
            conv2d(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_3',
                activation=None,
            ),
            self.activation,
            ResidualBlock(
                self.channels,
                kernel=[2, 2],
                activation=self.activation,
                use_batch_norm=False,
            ),
            conv2d(
                self.hidden,
                [2, 2],
                [2, 2],
                name='conv_4',
                activation=tf.nn.sigmoid,
            ),
        ]
        super().build(input_shape)


class Decoder(SummaryModel):

    def __init__(
        self,
        channels,
        activation='relu',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        if activation == 'relu':
            self.activation = tf.keras.layers.Activation('relu')
        elif activation == 'tanh':
            self.activation = tf.keras.layers.Activation('tanh')
        else:
            raise ValueError('unsupported activation')
        self.build(None)

    def build(self, input_shape):
        self.model_layers = [
            conv2d_transpose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_2',
                activation=None,
            ),
            self.activation,
            ResidualBlock(
                self.channels,
                kernel=[2, 2],
                activation=self.activation,
                use_batch_norm=False,
            ),
            conv2d_transpose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_3',
                activation=None,
            ),
            self.activation,
            ResidualBlock(
                self.channels,
                kernel=[2, 2],
                activation=self.activation,
                use_batch_norm=False,
            ),
            conv2d_transpose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_4',
                activation=None,
            ),
            self.activation,
            ResidualBlock(
                self.channels,
                kernel=[2, 2],
                activation=self.activation,
                use_batch_norm=False,
            ),
            conv2d_transpose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_5',
                activation=None,
            ),
            self.activation,
            conv2d_transpose(
                3,
                [1, 1],
                [1, 1],
                name='deconv_6',
                activation=None,
            ),
        ]
        super().build(input_shape)


class Upsampler(SummaryModel):

    def __init__(
        self,
        channels,
        activation='relu',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        if activation == 'relu':
            self.activation = tf.keras.layers.Activation('relu')
        elif activation == 'tanh':
            self.activation = tf.keras.layers.Activation('tanh')
        else:
            raise ValueError('unsupported activation')
        self.build(None)

    def build(self, input_shape):
        self.model_layers = [
            conv2d_transpose(
                self.channels,
                [2, 2],
                [2, 2],
                name='upsample_1',
                activation=None,
            ),
            self.activation,
            conv2d_transpose(
                3,
                [1, 1],
                [1, 1],
                name='upsample_2',
                activation=None,
            ),
            self.activation,
        ]
        super().build(input_shape)
