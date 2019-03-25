from __future__ import absolute_import

from global_variables import *
import main
import layers
import pprint
import tensorflow as tf


def define_additional_flags():
    flags = Namespace()
    flags.debug = 2
    flags.local = True
    define_flags(args=[], additional_flags=flags)


def test_decoder(capsys):
    with capsys.disabled():
        define_additional_flags()
        pprint.pprint(FLAGS.__dict__)
        input = tf.placeholder(tf.float32, [None, 4, 4, 32])
        layers.Decoder(32)(input)


def test_downsampler(capsys):
    with capsys.disabled():
        define_additional_flags()
        pprint.pprint(FLAGS.__dict__)
        input = tf.placeholder(tf.float32, [None, 4, 4, 3])
        output = layers.Downsampler(3)(input)
        print(output.shape)


# def test_build(capsys):
#     with capsys.disabled():
#         define_additional_flags()
#         main.construct_vae(tf.placeholder(tf.float32, [None, *DIM]), DIM)
#         main.print_params()
