from __future__ import absolute_import
import subprocess
import os
import types
import pprint
import argparse
import util

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


def enum_of(options):

    def check(val):
        if val in options:
            return val
        raise ValueError(f'enum option is not valid must be one of {options}')

    return check


FLAGS = Namespace(is_set=False)


def post_setup():
    FLAGS._revision = subprocess.run(
        'git rev-parse HEAD'.split(' '),
        stdout=subprocess.PIPE,
    ).stdout.decode('ascii').strip()

    commit_message = subprocess.run(
        'git log -n 1 HEAD'.split(' '),
        stdout=subprocess.PIPE,
    ).stdout.decode('ascii').strip()

    if FLAGS.debug >= 1:
        util.print_wrapper('commit', lambda: print(commit_message))

    if not FLAGS.local:
        import atexit
        import tensorflow as tf

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


def int_or_none(value):
    if value == 'None':
        return None
    else:
        return int(value)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def define_flags(additional_flags=None, modify_parser=None, args=None):
    reset = False    # @param {type: "boolean"}
    if (not reset) and FLAGS.is_set:
        return

    parser = argparse.ArgumentParser()

    if modify_parser is not None:
        modify_parser(parser)

    parser.add_argument(
        '-batch_size',
        default=16,
        type=int,
        help=f'the batch size, default: {16}'
    )

    parser.add_argument(
        '-categorical_dims',
        default=5,
        type=int,
    )

    parser.add_argument(
        '-channel_dims',
        default=64,
        type=int,
    )

    parser.add_argument(
        '-disable_residual_block',
        default=True,
        type=boolean_string,
    )

    parser.add_argument(
        '-epochs',
        default=1000,
        type=int,
    )

    parser.add_argument(
        '-hidden_dims',
        default=32,
        type=int,
    )

    parser.add_argument(
        '-is_set',
        default=True,
        type=boolean_string,
    )

    parser.add_argument(
        '-learning_rate',
        default=1e-3,
        type=float,
    )

    parser.add_argument(
        '-run_test',
        default=False,
        type=boolean_string,
    )

    parser.add_argument(
        '-auto_restore',
        default=False,
        type=boolean_string,
    )

    parser.add_argument(
        '-restore',
        default=None,
        type=str,
    )

    parser.add_argument(
        '-save_freq',
        default=5,
        type=int,
    )

    parser.add_argument(
        '-max_to_keep',
        default=20,
        type=int,
    )

    parser.add_argument(
        '-run_type',
        default='primary',
        type=str,
    )

    parser.add_argument(
        '-helper_mse_loss',
        default=False,
        type=boolean_string,
    )

    parser.add_argument(
        '-summarize',
        default=True,
        type=boolean_string,
    )

    parser.add_argument(
        '-summary_frequency',
        default=200,
        type=int,
    )

    parser.add_argument(
        '-tensorboard_port',
        default='8080',
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
        type=int,
    )

    parser.add_argument(
        '-holdout_size',
        default=200,
        type=int,
    )

    parser.add_argument(
        '-prefetch',
        default=40,
        type=int,
        help='amount to be fetched in advance',
    )

    parser.add_argument(
        '-gaussian_downsample',
        nargs='*',
        default=False,
        type=boolean_string,
        help='downsample with gaussian blur',
    )

    parser.add_argument(
        '-default_filter',
        nargs='*',
        default=[5, 5],
        type=int,
        help='default filter',
    )

    parser.add_argument(
        '-increment_size_intervals',
        nargs='*',
        default=[20, 50, 100],
        type=int,
        help='epoch at which the model doubles in size',
    )

    parser.add_argument(
        '-fixed_size',
        default=64,
        type=int_or_none,
        help='fixed crop size',
    )

    parser.add_argument(
        '-min_size',
        default=16,
        type=int,
        help='minimum pixel block width',
    )

    parser.add_argument(
        '-latent_gamma',
        default=0.0,
        type=float,
        help='the scale factor of how much that gradient is affected'
    )

    parser.add_argument(
        '-reuse',
        default=True,
        type=boolean_string,
        help='wether or not to reuse neural networks across dimensions',
    )

    parser.add_argument(
        '-progress',
        default=True,
        type=boolean_string,
        help='display epoch progress results',
    )

    parser.add_argument(
        '-use_ssim',
        default=False,
        type=boolean_string,
        help='use ssim based loss',
    )

    if hasattr(additional_flags, 'local'):
        default_local = additional_flags.local
    else:
        default_local = os.uname()[1] == 'XPS'

    default_bucket = 'gs://yiblet_research'

    parser.add_argument(
        '-local',
        default=default_local,
        type=boolean_string,
    )

    parser.add_argument(
        '-bucket',
        default=default_bucket,
        type=str,
    )

    if default_local:
        default_data = 'data/cifar10'
        default_large_image_dir = 'local/images'
        default_summaries_dir = 'local/saves'
        default_saves_dir = 'local/summaries'
        default_tf_records_dir = 'local/records'
        default_tpu_address = None
        print('running locally')
    else:
        print('remember to authenticate the user')

        default_data = '/gdrive/My Drive/cifar10'
        default_large_image_dir = f'{default_bucket}/images'
        default_summaries_dir = f'{default_bucket}/summaries'
        default_saves_dir = f'{default_bucket}/saves'
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
        '-gaussian_kernel_width',
        default=3,
        type=int,
    )

    parser.add_argument(
        '-use_batch_norm',
        default=True,
        type=boolean_string,
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
        '-quantize',
        default='quantize',
        type=enum_of([
            'quantize',
            'noise',
            'none',
        ]),
    )

    parser.add_argument(
        '-quantization_bits',
        default=8.0,
        type=float,
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
        '-saves_dir',
        default=default_saves_dir,
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
        type=boolean_string,
    )

    parser.parse_args(args=args, namespace=FLAGS)

    if additional_flags is not None:
        FLAGS.bulk_update(additional_flags)

    FLAGS.use_tpu = FLAGS.tpu_address is not None

    post_setup()


if __name__ == "__main__":
    define_flags()
    pprint.pprint(FLAGS.__dict__)
