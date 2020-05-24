#!/usr/bin/env python3
"""Script to create a batch normalization layer in a DNN
    using tensorflow"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Function that normalized a batch in a DNN with Tf
    Args:
        prev: the activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function that should be used
                    on the output of the layer

    Returns: tensor of the activated output for the layer

    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n, activation=None, kernel_initializer=init)
    x_prev = x(prev)
    scale = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    mean, variance = tf.nn.moments(x_prev, axes=[0])
    offset = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    variance_epsilon = 1e-8

    normalization = tf.nn.batch_normalization(
        x_prev,
        mean,
        variance,
        offset,
        scale,
        variance_epsilon,
    )
    return activation(normalization)
