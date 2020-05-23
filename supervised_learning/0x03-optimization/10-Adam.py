#!/usr/bin/env python3
"""Script to optimize DNN using Adam with Tensorflow"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Function to train a DNN with TF RMSProp optimization
    Args:
        loss: loss of the network
        alpha: learning rate
        beta1: weight used for the first moment
        beta2: weight used for the second moment
        epsilon: small number to avoid division by zero

    Returns: Adam optimization operation

    """
    optimizer = tf.train.AdamOptimizer(alpha,
                                       beta1,
                                       beta2,
                                       epsilon).minimize(loss)
    return optimizer
