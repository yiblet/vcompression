from __future__ import absolute_import
import numpy as np
import subprocess
import sys
import os
import types
import tensorflow as tf
import pprint
import argparse

WIDTH = 32
HEIGHT = WIDTH
CHANNEL = 3
DIM = (HEIGHT, WIDTH, CHANNEL)
DEFAULT_SUMMARY_COLLECTION = 'summaries'


class Namespace(types.SimpleNamespace):
    """Docstring for Namespace. """

    def __init__(self, **kwargs):
        types.SimpleNamespace.__init__(self, **kwargs)

    def __add__(self, other):
        if (isinstance(other, types.SimpleNamespace)):
            return Namespace(**self.__dict__, **other.__dict__)
        else:
            raise ValueError("not correct type")

    def bulk_update(self, other):
        if (isinstance(other, types.SimpleNamespace)):
            for k, v in other.__dict__.items():
                setattr(self, k, v)


FLAGS = Namespace(is_set=False)


def run_subprocesses():
    if not FLAGS.local:
        import atexit

        if 'COLAB_TPU_ADDR' in os.environ:
            FLAGS.tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']

            print('TPU address is', FLAGS.tpu_address)

            with tf.Session(FLAGS.tpu_address) as session:
                devices = session.list_devices()

            print('TPU devices:')
            pprint.pprint(devices)

        if FLAGS.run_subprocesses:
            processes = [
                subprocess.Popen(
                    "kill $(ps -A | grep tensorboard | grep -o '^[0-9]\\+')",
                    shell=True,
                    stdout=subprocess.PIPE,
                    stdin=subprocess.PIPE
                ),
                subprocess.Popen(
                    f"python3 scripts/logger.py \
                        '{FLAGS.summaries_dir}' \
                        --port {FLAGS.tensorboard_port}",
                    shell=True,
                ),
            ]

            def kill_subprocesses():
                for process in processes:
                    process.kill()

            atexit.register(kill_subprocesses)
            print('goto: https://tensor.serveo.net to view logs')


def define_flags(additional_flags=None):
    reset = False    # @param {type: "boolean"}
    if (not reset) and FLAGS.is_set:
        return

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-batch_size',
        default=16,
        type=str,
        help=f'the batch size, default: {16}'
    )
    parser.add_argument(
        '-categorical_dims',
        default=5,
        type=str,
    )
    parser.add_argument(
        '-channel_dims',
        default=64,
        type=str,
    )
    parser.add_argument(
        '-disable_residual_block',
        default=True,
        type=bool,
    )
    parser.add_argument(
        '-epochs',
        default=1000,
        type=str,
    )
    parser.add_argument(
        '-hidden_dims',
        default=32,
        type=str,
    )
    parser.add_argument(
        '-is_set',
        default=True,
        type=bool,
    )
    parser.add_argument(
        '-learning_rate',
        default=1e-3,
        type=str,
    )
    parser.add_argument(
        '-run_type',
        default='primary',
        type=str,
    )
    parser.add_argument(
        '-summarize',
        default=True,
        type=bool,
    )
    parser.add_argument(
        '-summary_frequency',
        default=200,
        type=str,
    )
    parser.add_argument(
        '-tensorboard_port',
        default=8080,
        type=str,
    )
    parser.add_argument(
        '-test_dir',
        default='test',
        type=str,
    )
    parser.add_argument(
        '-train_dir',
        default='train',
        type=str,
    )
    parser.add_argument(
        '-train_steps',
        default=600,
        type=str,
    )
    parser.add_argument(
        '-tunnel_loc',
        default='yiblet',
        type=str,
    )
    parser.add_argument(
        '-holdout_size',
        default=200,
        type=str,
    )
    parser.add_argument(
        '-prefetch',
        default=40,
        type=int,
        help='amount to be fetched in advance',
    )
    parser.add_argument(
        '-crop_size',
        default=32,
        type=str,
    )

    if hasattr(additional_flags, 'local'):
        default_local = additional_flags.local
    else:
        default_local = os.uname()[1] == 'XPS'

    default_bucket = 'gs://yiblet_research'

    parser.add_argument(
        '-local',
        default=default_local,
        type=bool,
    )

    parser.add_argument(
        '-bucket',
        default=default_bucket,
        type=str,
    )

    if default_local:
        default_data = 'data/cifar10'
        default_large_image_dir = 'local/images'
        default_summaries_dir = 'local/summaries'
        default_tf_records_dir = 'local/records'
        default_tpu_address = None
        print('running locally')
    else:
        print('mounting google drive')
        from google.colab import auth
        auth.authenticate_user()

        default_data = '/gdrive/My Drive/cifar10'
        default_large_image_dir = f'{default_bucket}/images'
        default_summaries_dir = f'{default_bucket}/summaries'
        default_tf_records_dir = default_data
        default_tpu_address = None

    # defaults that are the same for both types
    default_directory = 'out'
    default_debug = 0

    parser.add_argument(
        '-record_data',
        default=f'{default_bucket}/test.tfrecord',
        type=str,
    )

    parser.add_argument(
        '-data',
        default=default_data,
        type=str,
    )

    parser.add_argument(
        '-debug',
        default=default_debug,
        type=int,
    )

    parser.add_argument(
        '-directory',
        default=default_directory,
        type=str,
    )

    parser.add_argument(
        '-large_image_dir',
        default=default_large_image_dir,
        type=str,
        help='location of the images'
    )

    parser.add_argument(
        '-summaries_dir',
        default=default_summaries_dir,
        type=str,
    )

    parser.add_argument(
        '-tf_records_dir',
        default=default_tf_records_dir,
        type=str,
    )

    parser.add_argument(
        '-tpu_address',
        default=default_tpu_address,
        type=str,
    )

    parser.add_argument(
        '-run_subprocesses',
        default=False,
        type=bool,
    )

    parser.parse_args(namespace=FLAGS)

    if additional_flags is not None:
        FLAGS.bulk_update(additional_flags)

    FLAGS.use_tpu = FLAGS.tpu_address is not None

    run_subprocesses()


if __name__ == "__main__":
    define_flags()
    pprint.pprint(FLAGS.__dict__)
