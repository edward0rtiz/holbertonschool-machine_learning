#!/usr/bin/env python3
"""Script to implement learning rate decay
    hyperparameter"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    function of learning rate decay in a DNN"
    Args:
        alpha: original learning rate
        decay_rate: weight used to determine the rate at which alpha will decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes of gradient descent that should occur
                    before alpha is decayed further
    Returns: updated value for alpha

    """
    alpha = alpha / (1 + decay_rate * int(global_step / decay_step))
    return alpha
