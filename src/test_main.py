from __future__ import absolute_import

from global_variables import *
import main


def define_additional_flags():
    FLAGS = Namespace()
    FLAGS.debug = True
    FLAGS.local = True
    main.define_flags(additional_flags=FLAGS)


def test_build(capsys):
    with capsys.disabled():
        define_additional_flags()
        main.model_fn(
            tf.placeholder(tf.float32, [None, *DIM]),
            tf.placeholder(tf.float32, [None, *DIM]),
            tf.estimator.ModeKeys.EVAL,
            params=main.construct_params(channel=64),
        )
