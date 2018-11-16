import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

tf.enable_eager_execution()

categorical = tf.get_variable(
    name='categorical_distribution',
    shape=[3],
)
categorical = tf.nn.softmax(categorical)
loc = tf.get_variable(
    name='logistic_loc_variables',
    shape=[3],
)
scale = tf.nn.softplus(tf.get_variable(
    name='logistic_scale_variables',
    shape=[3],
))

mixture = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(
        probs=categorical
    ),
    components_distribution=tfd.Logistic(
        loc=loc,
        scale=scale,
    ),
)
