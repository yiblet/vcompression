from __future__ import absolute_import
from global_variables import *
import summary
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

# ------------------------------------------------------------------------------
# ------------------------------- utility layers -------------------------------
# ------------------------------------------------------------------------------


class Identity(tf.keras.layers.Layer):

    def call(self, input):
        return input


class HasBatchNorm(object):

    def __init__(self, training):
        self.training = training

    def batch_norm(self, **kwargs):
        if FLAGS.use_batch_norm:
            layer = tf.layers.BatchNormalization(**kwargs)
            return lambda x: layer(x, training=self.training)
        else:
            return Identity()


class HasActivation(object):

    def __init__(self, activation_type):
        self.activation_type = activation_type

    def activation(self):
        if self.activation_type is not None:
            return tf.keras.layers.Activation(self.activation_type)
        else:
            # no activation return the identity layer
            return Identity()


@tf.custom_gradient
def scale_gradient(input):

    def grad(dy):
        return dy * FLAGS.latent_gamma

    return input, grad


# ------------------------------------------------------------------------------
# ----------------------------- the important bits -----------------------------
# ------------------------------------------------------------------------------


class Downsampler(tf.keras.layers.Layer):

    def __init__(self, size, **kwargs):
        tf.keras.layers.Layer.__init__(self, **kwargs)
        self.size = size

    def build(self, input_shape):

        with summary.SummaryScope(self.name) as scope:
            self.std = self.add_weight(
                'std',
                shape=[int(input_shape[-1])],
                trainable=True,
            )
            scale = tf.nn.relu(self.std) + 1e-6

            dist = tfp.distributions.Normal(
                loc=tf.zeros_like(scale), scale=scale
            )
            vals = dist.prob(
                tf.range(
                    start=-self.size,
                    limit=self.size + 1,
                    dtype=tf.float32,
                )[:, tf.newaxis]
            )

            gauss_kernel = vals[:, tf.newaxis, :] * vals[tf.newaxis, :, :]
            gauss_kernel /= tf.reduce_sum(gauss_kernel, axis=[0, 1])
            self.gauss_kernel = gauss_kernel[:, :, :, tf.newaxis]

            scope['std'] = self.std

    def call(self, input):

        return tf.nn.depthwise_conv2d(
            input,
            self.gauss_kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )[:, ::2, ::2, :]


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
                shape=[FLAGS.hidden_dims, FLAGS.categorical_dims],
                trainable=True,
            )
            categorical = tf.nn.softmax(categorical)
            loc = self.add_weight(
                name='logistic_loc_variables',
                shape=[FLAGS.hidden_dims, FLAGS.categorical_dims],
                trainable=True,
            )
            scale = tf.nn.softplus(
                self.add_weight(
                    name='logistic_scale_variables',
                    shape=[FLAGS.hidden_dims, FLAGS.categorical_dims],
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

        if FLAGS.latent_gamma == 0.0:
            stopped_latents = tf.stop_gradient(latent)
        else:
            stopped_latents = scale_gradient(latent)

        likelihoods = self._distribution.cdf(
            tf.clip_by_value(stopped_latents + 0.5 / 255.0, -1.0, 1.0)
        ) - self._distribution.cdf(
            tf.clip_by_value(stopped_latents - 0.5 / 255.0, -1.0, 1.0)
        )

        return likelihoods


class ResidualBlock(tf.keras.Model, HasBatchNorm, HasActivation):
    """Basic Residual Block"""

    def __init__(
        self,
        channels,
        training,
        kernel=[3, 3],
        activation=None,
        batch_norm_last=True,
        trainable=True,
        **kwargs
    ):
        # multiple inheritance initialization
        tf.keras.Model.__init__(self, **kwargs)
        HasBatchNorm.__init__(self, training)
        HasActivation.__init__(self, activation)

        self.trainable = True
        self.batch_norm_last = batch_norm_last
        self.channels = channels
        self.kernel = kernel
        self.init()

    def init(self):
        if FLAGS.disable_residual_block:
            return

        create_batch_norm = self.batch_norm()

        last_batch_norm = None
        if self.batch_norm_last:
            last_batch_norm = self.batch_norm()

        with tf.name_scope(self.name):
            self.model_layers = [
                tf.layers.Conv2D(
                    self.channels,
                    self.kernel,
                    dilation_rate=[2, 2],
                    name="conv2d_1",
                    activation=None,
                    padding="same",
                    trainable=self.trainable,
                ),
                create_batch_norm,
                self.activation(),
                tf.layers.Conv2D(
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

        res = self.activation()(cur + input)
        # renaming output to residual_block
        return tf.identity(res, name=self.name)


class SummaryModel(tf.keras.Model):

    def __init__(
        self,
        silent=False,
        **kwargs,
    ):
        tf.keras.Model.__init__(self, **kwargs)
        self.silent = silent

    def build(self, _):
        pass

    def call(self, input):
        with summary.SummaryScope(self.name) as scope:
            scope['input'] = input
            output = scope.sequential(input, self.model_layers)
        return output


class Encoder(SummaryModel, HasBatchNorm, HasActivation):

    def __init__(
        self,
        channels,
        hidden,
        training,
        activation='relu',
        **kwargs,
    ):
        # multiple inheritance initialization
        SummaryModel.__init__(self, **kwargs)
        HasBatchNorm.__init__(self, training)
        HasActivation.__init__(self, activation)

        self.channels = channels
        self.hidden = hidden
        self.build(None)

    def build(self, input_shape):
        self.model_layers = [
            tf.layers.Conv2D(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_1',
                activation=None,
            ),
            self.batch_norm(),
            self.activation(),
            ResidualBlock(
                self.channels,
                self.training,
                kernel=[2, 2],
                activation=self.activation_type,
            ),
            tf.layers.Conv2D(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_2',
                activation=None,
            ),
            self.batch_norm(),
            self.activation(),
            ResidualBlock(
                self.channels,
                self.training,
                kernel=[2, 2],
                activation=self.activation_type,
            ),
            tf.layers.Conv2D(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_3',
                activation=None,
            ),
            self.batch_norm(),
            self.activation(),
            ResidualBlock(
                self.channels,
                self.training,
                kernel=[2, 2],
                activation=self.activation_type,
            ),
            tf.layers.Conv2D(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_4',
                activation=None,
            ),
            self.batch_norm(),
            self.activation(),
            ResidualBlock(
                self.channels,
                self.training,
                kernel=[2, 2],
                activation=self.activation_type,
            ),
            tf.layers.Conv2D(
                self.hidden,
                [2, 2],
                [1, 1],
                padding='same',
                name='conv_5',
                activation=tf.nn.sigmoid,
            ),
        ]


class Decoder(SummaryModel, HasBatchNorm, HasActivation):

    def __init__(
        self,
        channels,
        training,
        activation='relu',
        **kwargs,
    ):
        # multiple inheritance initialization
        SummaryModel.__init__(self, **kwargs)
        HasBatchNorm.__init__(self, training)
        HasActivation.__init__(self, activation)

        self.channels = channels
        self.init()

    def init(self):
        self.model_last = tf.layers.Conv2DTranspose(
            3,
            [2, 2],
            [1, 1],
            name='deconv_to_image',
            activation=None,
            padding='same',
        )

        self.model_layers = [
            tf.layers.Conv2DTranspose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_2',
                activation=None,
            ),
            self.batch_norm(),
            self.activation(),
            ResidualBlock(
                self.channels,
                self.training,
                kernel=[2, 2],
                activation=self.activation_type,
            ),
            tf.layers.Conv2DTranspose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_3',
                activation=None,
            ),
            self.batch_norm(),
            self.activation(),
            ResidualBlock(
                self.channels,
                self.training,
                kernel=[2, 2],
                activation=self.activation_type,
            ),
            tf.layers.Conv2DTranspose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_4',
                activation=None,
            ),
            self.batch_norm(),
            self.activation(),
            ResidualBlock(
                self.channels,
                self.training,
                kernel=[2, 2],
                activation=self.activation_type,
            ),
            tf.layers.Conv2DTranspose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_5',
                activation=None,
            ),
            self.batch_norm(),
            self.activation(),
            ResidualBlock(
                self.channels,
                self.training,
                kernel=[2, 2],
                activation=self.activation_type,
            ),
            self.model_last,
        ]

        self.model_upsize = [
            tf.layers.Conv2DTranspose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_upsize',
                activation=None,
            ),
            self.batch_norm(),
            self.activation(),
            ResidualBlock(
                self.channels,
                self.training,
                kernel=[2, 2],
                activation=self.activation_type,
            ),
            self.model_last,
        ]

    def call(self, input):
        with summary.SummaryScope(self.name) as scope:
            scope['input'] = input
            layers = scope.sequential(
                input,
                self.model_layers,
                interior_layers=True,
            )
            output = layers[-1]
            internal = layers[-2]

        with summary.SummaryScope(self.name + "_upsize") as scope:
            upsize = scope.sequential(internal, self.model_upsize)

        return output, upsize
