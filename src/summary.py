from __future__ import absolute_import
import os
import pprint
import tensorflow as tf
import numpy as np
from global_variables import *

DEFAULT_SUMMARY_COLLECTION = 'summary'


def variable_summaries(key, var, collection=DEFAULT_SUMMARY_COLLECTION):
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


class Summarize(tf.keras.layers.Layer):

    def __init__(self, key=None, **kwargs):
        super().__init__(**kwargs)
        self.key = key or self.name

    def call(self, input):
        variable_summaries(self.key, input)

        if FLAGS.debug >= 1:
            print(f'{self.key}: {input.shape}')

        return input


class SummaryScope(dict):
    '''wrapper around tf.name_scope
       that also logs all tensors that are put into the scope dict'''

    def __init__(
        self,
        scope_name,
        silent=False,
        collection=DEFAULT_SUMMARY_COLLECTION,
    ):
        super()
        self.scope_name = scope_name
        self.collection = collection
        self.silent = silent
        self.is_sequential = False

    def _get_name(self, name):
        name = name[:name.rindex('/')]
        name = name[name.rindex('/') + 1:]
        return name

    def sequential(
        self,
        input,
        ops,
        include_input=False,
        interior_layers=False,
    ):
        prev_op = input

        if FLAGS.debug >= 2:
            print(f'printing {self.scope_name} dimensions')

        if include_input:
            name = self._get_name(prev_op.name)
            if FLAGS.debug >= 2:
                print(f'{name}: {prev_op.shape}')
            self[name] = prev_op

        layers = []

        for operation in ops:
            prev_prev_op = prev_op
            prev_op = operation(prev_op)
            if prev_prev_op == prev_op:
                continue

            layers.append(prev_op)
            name = self._get_name(prev_op.name)
            if FLAGS.debug >= 2:
                print(f'{name}: {prev_op.shape}')
            self[name] = prev_op

        self.is_sequential = True

        if FLAGS.debug >= 2:
            print('')

        return layers if interior_layers else prev_op

    def __enter__(self):
        self.scope = tf.name_scope(self.scope_name)
        self.scope.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        self.scope.__exit__(type, value, traceback)

        if (not self.is_sequential) and FLAGS.debug >= 2:
            print(f'printing {self.scope_name} dimensions')
            for k, v in self.items():
                print(f'{k}: {v.shape}')
            print('')

        if not self.silent:
            for (key, var) in self.items():
                variable_summaries(key, var, self.collection)
