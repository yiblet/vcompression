from __future__ import absolute_import
import os
import pprint
import tensorflow as tf
import numpy as np
from global_variables import *


def variable_summaries(key, var, collection):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""

    with tf.name_scope(f'{key}_summaries'):
        if not FLAGS.summarize:
            tf.summary.histogram('histogram', var)
        else:
            mean = tf.reduce_mean(var)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.scalar('stddev', stddev)
            tf.summary.histogram('histogram', var)


class SummaryScope(dict):
    def __init__(self, scope_name, collection=DEFAULT_SUMMARY_COLLECTION):
        super()
        self.scope_name = scope_name
        self.collection = collection

    def _get_name(self, name):
        name = name[:name.rindex('/')]
        name = name[name.rindex('/') + 1:]
        return name

    def sequential(self, input, ops, include_input=False):
        prev_op = input

        x = str("")

        if include_input:
            self[self._get_name(prev_op.name)] = prev_op

        for operation in ops:
            prev_op = operation(prev_op)
            name = self._get_name(prev_op.name)
            self[name] = prev_op
            new_op = True

        return prev_op

    def __enter__(self):
        self.scope = tf.name_scope(self.scope_name)
        self.scope.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.scope.__exit__(type, value, traceback)

        if FLAGS.debug:
            print(f'printing {self.scope_name} dimensions')
            for k, v in self.items():
                print(f'{k}: {v.shape}')
            print('')

        for (key, var) in self.items():
            variable_summaries(key, var, self.collection)
