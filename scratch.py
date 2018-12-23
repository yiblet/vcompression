#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sys
import os

tfd = tfp.distributions

tf.enable_eager_execution()

LARGE_IMAGE_DIR = 'local/images'

images = [f'{LARGE_IMAGE_DIR}/{image}' for image in os.listdir(LARGE_IMAGE_DIR)]

dataset = tf.data.Dataset.from_tensor_slices(
    tf.io.matching_files(f'{LARGE_IMAGE_DIR}/*jpg')
)

dataset = dataset.map(parse_image).filter(is_large_image).map(reshape).take(2)
print(list(dataset))
