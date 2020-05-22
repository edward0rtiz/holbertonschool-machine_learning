#!/usr/bin/env python3
"""Script of momentum in Tensorflow"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Function to train a DNN with TF momentum optimization
    Args:
        loss: loss of the network
        alpha: learning rate
        beta1: momentum weight

    Returns: momentum optimization operation

    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return optimizer
