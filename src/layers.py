import tensorflow as tf


# class Compressor(tf.keras.layers.Layer):

#     """Docstring for Compressor. """

#     def __init__(
#         self,
#         numeric_range=(-1, 1),
#         name=name,
#     ):
#         """TODO: to be defined1. """
#         tf.keras.layers.Layer.__init__(self, name=name)
#         self.numeric_range = numeric_range

#     def build(self, input_shape):
#         if len(input_shape != 4):
#             raise ValueError('input needs to have rank 4')


class ResidualBlock(tf.keras.layers.Layer):
    """Basic Residual Block"""

    def __init__(
        self,
        channels,
        kernel=[3, 3],
        activation=None,
        name=None,
        batch_norm_last=True,
    ):
        tf.keras.layers.Layer.__init__(self, name=name)
        self.activation = activation
        self.batch_norm_last = batch_norm_last
        self.channels = channels
        self.kernel = [3, 3]

    def build(self, input_shape):

        if len(input_shape) != 4:
            raise ValueError("input needs to have rank 4")

        activation = self.activation

        if activation is None:
            activation = [None, None, None]
        elif not isinstance(activation, list):
            activation = [activation, activation, activation]

        last_batch_norm = None
        if not self.batch_norm_last:
            last_batch_norm = tf.layers.BatchNormalization()

        with tf.name_scope(self.name):
            self.layers = [
                tf.layers.Conv2D(
                    self.channels,
                    [1, 1],
                    [1, 1],
                    name="conv2d_1",
                    activation=None,
                    padding="same",
                ),
                tf.layers.BatchNormalization(),
                activation[0],
                tf.layers.Conv2D(
                    self.channels,
                    self.kernel,
                    [1, 1],
                    name="conv2d_2",
                    activation=None,
                    padding="same",
                ),
                tf.layers.BatchNormalization(),
                activation[1],
                tf.layers.Conv2D(
                    self.channels,
                    [1, 1],
                    [1, 1],
                    name="conv2d_3",
                    activation=None,
                    padding="same",
                ),
                last_batch_norm,
                activation[2],
            ]

    def call(self, input):
        cur = input
        for layer in self.layers:
            if layer is not None:
                cur = layer(cur)
        return cur + input
