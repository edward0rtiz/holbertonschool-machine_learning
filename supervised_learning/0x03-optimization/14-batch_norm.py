#!/usr/bin/env python3
"""Script to create a batch normalization layer in a DNN
    using tensorflow"""

import numpy as np
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Function that normalized a batch in a DNN with Tf
    Args:
        prev: the activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function that should be used on the output of the layer

    Returns: tensor of the activated output for the layer

    """
    x = tf.layers.Dense(units=n, activation=None)
    f = tf.layers.batch_normalization(inputs=x(prev))
    return activation(f)
