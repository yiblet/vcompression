from __future__ import absolute_import
import numpy as np
import subprocess
import sys
import os
import types
import tensorflow as tf
import pprint


WIDTH = 32
HEIGHT = WIDTH
CHANNEL = 3
DIM = (HEIGHT, WIDTH, CHANNEL)
URL_LOG = 'url.txt'
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
        if 'COLAB_TPU_ADDR' in os.environ:
            FLAGS.tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']

            print('TPU address is', FLAGS.tpu_address)

            with tf.Session(FLAGS.tpu_address) as session:
                devices = session.list_devices()

            print('TPU devices:')
            pprint.pprint(devices)

        subprocess.Popen(
            "kill $(ps -A | grep tensorboard | grep -o '^[0-9]\\+')",
            shell=True,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
        ) .communicate()

        subprocess.Popen(
            "kill $(ps -A | grep lt | grep -o '^[0-9]\\+')",
            shell=True,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
        ) .communicate()

        subprocess.Popen(
            f"rm '{URL_LOG}'",
            shell=True,
        )

        print(subprocess.Popen(
            f"npm install -g localtunnel; lt --port {FLAGS.tensorboard_port} -s {FLAGS.tunnel_loc} > {URL_LOG} 2>&1 &",
            shell=True,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE
        ).communicate()[0].decode('ascii'))

        subprocess.Popen(
            f"rm -r '{FLAGS.summaries_dir}'",
            shell=True,
        ).wait()

        subprocess.Popen(
            f"tensorboard --logdir '{FLAGS.summaries_dir}' --port {FLAGS.tensorboard_port} --host 0.0.0.0 >> tensorboard.log 2>&1 &",
            shell=True,
        )


def define_flags(additional_flags=None):
    reset = False  # @param {type: "boolean"}
    if (not reset) and FLAGS.is_set:
        return

    FLAGS.categorical_dims = 3
    FLAGS.batch_size = 16  # @param {type: "number"}
    FLAGS.epochs = 1000  # @param {type: "number"}
    FLAGS.is_set = True
    FLAGS.learning_rate = 1e-3  # @param {type: "number"}
    FLAGS.summary_frequency = 200  # @param {type: "number"}
    FLAGS.train_steps = 600  # @param {type: "number"}
    FLAGS.z_dims = 128  # @param {type: "number"}
    FLAGS.summarize = True
    FLAGS.local = os.uname()[1] == 'XPS'
    FLAGS.disable_residual_block = True
    FLAGS.tunnel_loc = 'yiblet'  # @param
    FLAGS.tensorboard_port = 8080  # @param {type : "number"}

    if FLAGS.local:
        FLAGS.data = 'data/cifar10'
        FLAGS.debug = True
        FLAGS.directory = 'out'
        FLAGS.summaries_dir = 'local/summaries'
        FLAGS.tpu_address = None

        print('running locally')

    else:
        print('mounting google drive')
        from google.colab import drive
        drive.mount('/gdrive')

        FLAGS.data = '/gdrive/My Drive/cifar10'
        FLAGS.debug = False
        FLAGS.directory = '/gdrive/My Drive/data_mnist'
        summaries_dir = 'summaries'  # @param {type: "string"}
        FLAGS.summaries_dir = f'/gdrive/My Drive/{summaries_dir}'
        FLAGS.tpu_address = None

    if additional_flags is not None:
        FLAGS.bulk_update(additional_flags)

    run_subprocesses()
