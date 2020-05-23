#!/usr/bin/env python3
"""Script to optimize DNN using RMSprop with Tensorflow"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Function to train a DNN with TF RMSProp optimization
    Args:
        loss: loss of the network
        alpha: learning rate
        beta2: RMSProp weight
        epsilon: small number to avoid division by zero

    Returns: momentum optimization operation

    """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2,
                                          epsilon=epsilon).minimize(loss)
    return optimizer
