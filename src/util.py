import tensorflow as tf


def compress(arg):
    placeholder = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

    channels = {
        'input': 3,
        'h1': 18,
        'h2': 1,
        'h3': 6,
        'output': 3,
    }

    variables = {
        'encoder_h1': tf.Variable(tf.truncated_normal(
            [5, 5, channels['input'], channels['h1']],
            stddev=0.01)
        ),
        'encoder_h2': tf.Variable(tf.truncated_normal(
            [5, 5, channels['h1'], channels['h2']], stddev=0.01)
        ),
        'decoder_h3': tf.Variable(tf.truncated_normal(
            [5, 5, channels['h2'], channels['h3']], stddev=0.01)
        ),
        'decoder_h4': tf.Variable(tf.truncated_normal(
            [5, 5, channels['h3'], channels['output']], stddev=0.01)
        ),
    }

    biases = {
        'encoder_h1_biases': tf.Variable(tf.truncated_normal(
            [channels['h1']],
            stddev=0.01)
        ),
        'encoder_h2_biases': tf.Variable(tf.truncated_normal(
            [channels['h2']], stddev=0.01)
        ),
        'decoder_h3_biases': tf.Variable(tf.truncated_normal(
            [channels['h3']], stddev=0.01)
        ),
        'decoder_h4_biases': tf.Variable(tf.truncated_normal(
            [channels['output']], stddev=0.01)
        ),
    }

    output = model(placeholder)
    print(output)
    pass


def contract(images):
    if get_image_dims(images)[0] == 2:
        return [images]
    smaller_images = resize(images)
    contracted_images = contract(smaller_images)
    smaller_images = contracted_images[-1]
    compress(images - upsize_images(smaller_images))


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
