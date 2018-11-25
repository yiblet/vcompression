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
        main.main()
