#!/usr/bin/env python3
"""Script to create a layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    method to create a TF layer
    Args:
        prev: tensor of the previous layer
        n: n nodes created
        activation: activation function

    Returns: Layer created with shape n

    """
    # Average number of inputs and output connections.
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode='FAN_AVG')
    layer = (tf.layers.Dense(units=n, activation=activation,
                             kernel_initializer=initializer,
                             name='layer'))
    return layer(prev)
