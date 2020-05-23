#!/usr/bin/env python3
"""Script to implement learning rate decay in DNN with Tensorflow"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    learning rate decay operation in tensorflow using inverse time decay:
    Args:
        alpha: original learning rate
        decay_rate: weight used to determine the rate at which alpha will decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes of gradient descent that should occur
                    before alpha is decayed further

    Returns:  learning rate decay operation

    """
    LRD = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                      decay_rate, staircase=True)
    return LRD
