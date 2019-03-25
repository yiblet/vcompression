from __future__ import absolute_import

from global_variables import *
import main
import layers
import pprint
import tensorflow as tf


def define_additional_flags(extra_flags=None):
    if extra_flags is None:
        flags = Namespace()
    else:
        flags = extra_flags
    flags.debug = 2
    flags.local = True
    define_flags(args=[], additional_flags=flags)


# ------------------------------------------------------------------------------
# ------------------------------- testing layers -------------------------------
# ------------------------------------------------------------------------------
#
def test_encoder(capsys):
    define_additional_flags()
    input = tf.placeholder(tf.float32, [None, 64, 64, 32])
    output = layers.Encoder(32, FLAGS.hidden_dims, False)(input)
    with capsys.disabled():
        print('encoder output:')
        print(output.shape)
        print('')


def test_encoder_with_resblock(capsys):
    define_additional_flags(Namespace(disable_residual_block=False))
    input = tf.placeholder(tf.float32, [None, 64, 64, 32])
    output = layers.Encoder(32, FLAGS.hidden_dims, False)(input)
    with capsys.disabled():
        print('encoder with residual blocks output:')
        print(output.shape)
        print('')


def test_decoder(capsys):
    define_additional_flags()
    input = tf.placeholder(tf.float32, [None, 4, 4, 32])
    outputs = layers.Decoder(32, False)(input)
    with capsys.disabled():
        print('decoder output:')
        for output in list(outputs):
            print(output.shape)
        print('')


def test_decoder_residual_block(capsys):
    define_additional_flags(Namespace(disable_residual_block=False))
    input = tf.placeholder(tf.float32, [None, 4, 4, 32])
    outputs = layers.Decoder(32, False)(input)
    with capsys.disabled():
        print('decoder with resblocks output:')
        for output in list(outputs):
            print(output.shape)
        print('')


def test_downsampler(capsys):
    define_additional_flags()
    input = tf.placeholder(tf.float32, [None, 4, 4, 3])
    output = layers.Downsampler(3)(input)
    with capsys.disabled():
        print('downsampler output:')
        print(output.shape)
        print('')
