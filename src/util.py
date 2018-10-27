import tensorflow as tf


def conv2d(x, W, b, strides=1):
    # W: [A, B, in_channels, out_channels]
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def deconv2d(x, W, b, out_shape, strides=1):
    # W: [A, B, in_channels, out_channels]
    x = tf.nn.conv2d_transpose(
        x,
        W,
        out_shape,
        strides=[1, strides, strides, 1],
        padding='SAME'
    )
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def get_dims(tensor):
    return [x.value for x in tf.shape(tensor)]


def get_image_dims(tensor):
    height_idx = 1
    width_idx = 2
    return get_dims(images)[height_idx:width_idx + 1]


def downsize(images, height_factor=2, width_factor=2):
    width, height = get_image_dims(images)
    return tf.image.resize_images(  # resize_images is possibly broken
        images,
        [height / height_factor, width / width_factor],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )


def upsize(images, height_factor=2, width_factor=2):
    width, height = get_image_dims(images)
    return tf.image.resize_images(  # resize_images is possibly broken
        images,
        [height * height_factor, width * width_factor],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )


def NAC(x, shape):
    with tf.variable_scope('nac'):
        w_hat = tf.get_variable("w_hat", shape)
        m_hat = tf.get_variable("h_hat", shape)
        w = tf.nn.tanh(w_hat) * tf.nn.sigmoid(m_hat)
        return (tf.matmul(x, w), w)
