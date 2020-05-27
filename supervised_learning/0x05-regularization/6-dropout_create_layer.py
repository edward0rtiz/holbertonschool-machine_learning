#!/usr/bin/env python3
"""Script to implement dropout in tensorflow"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Function that uses dropout in tensorflow
    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function that should be used on the layer
        keep_prob: probability that a node will be kept

    Returns: output of the new layer

    """
    dropout = tf.layers.Dropout(keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    tensor = tf.layers.Dense(units=n, activation=activation,
                             kernel_initializer=init,
                             kernel_regularizer=dropout)
    return tensor(prev)
