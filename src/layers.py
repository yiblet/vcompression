from __future__ import absolute_import
from global_variables import *
import summary
import tensorflow as tf
import tensorflow_probability as tfp


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

        stopped_latents = tf.stop_gradient(latent)
        likelihoods = self._distribution.cdf(
            tf.clip_by_value(stopped_latents + 0.5 / 255.0, 0.0, 1.0)
        ) - self._distribution.cdf(
            tf.clip_by_value(stopped_latents - 0.5 / 255.0, 0.0, 1.0)
        )

        return tf.log(likelihoods, name='bpp')


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
                self.activation[0],
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
            tf.layers.Conv2D(
                self.channels, [2, 2], [2, 2], name='conv_1', activation=None
            ),
            self.activation,
            ResidualBlock(
                self.channels, kernel=[3, 3], activation=self.activation
            ),
            tf.layers.Conv2D(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_2',
                activation=None,
            ),
            self.activation,
            ResidualBlock(
                self.channels, kernel=[3, 3], activation=self.activation
            ),
            tf.layers.Conv2D(
                self.channels,
                [2, 2],
                [2, 2],
                name='conv_3',
                activation=None,
            ),
            self.activation,
            ResidualBlock(
                self.channels, kernel=[3, 3], activation=self.activation
            ),
            tf.layers.Conv2D(
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
            tf.layers.Conv2DTranspose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_2',
                activation=None,
            ), self.activation,
            ResidualBlock(
                self.channels, kernel=[3, 3], activation=self.activation
            ),
            tf.layers.Conv2DTranspose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_3',
                activation=None,
            ), self.activation,
            ResidualBlock(
                self.channels, kernel=[3, 3], activation=self.activation
            ),
            tf.layers.Conv2DTranspose(
                self.channels,
                [2, 2],
                [2, 2],
                name='deconv_4',
                activation=None,
            ), self.activation,
            ResidualBlock(
                self.channels, kernel=[3, 3], activation=self.activation
            ),
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
