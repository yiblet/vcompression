import os
import pprint
import numpy as np
import time
import subprocess
import sys
import signal
import argparse

os.setpgrp()    # create new process group, become its leader

parser = argparse.ArgumentParser('logger script')

interrupted = False


def signal_handler(signal, frame):
    global interrupted
    interrupted = True


signal.signal(signal.SIGINT, signal_handler)

parser.add_argument(
    'logdir',
    type=str,
)

parser.add_argument(
    '--port',
    default=9001,
    type=int,
)

FLAGS = parser.parse_args()

process = subprocess.Popen(
    f'tensorboard --logdir {FLAGS.logdir} --host 0.0.0.0 --port {FLAGS.port}',
    shell=True,
)

signal.signal(signal.SIGINT, signal_handler)
while not interrupted:
    process = subprocess.Popen(
        f'ssh -o ServerAliveInterval=60 -o StrictHostKeyChecking=no -R tensor:80:localhost:{FLAGS.port} serveo.net',
        shell=True,
    )
    process.communicate()
    if process.returncode == 255 or process.returncode < 0:
        interrupted = True

os.killpg(0, signal.SIGKILL)
