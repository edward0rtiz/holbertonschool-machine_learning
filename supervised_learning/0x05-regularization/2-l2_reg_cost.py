#!/usr/bin/env python3
"""L2 regularization in tensorflow"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    Function to calculate the L2 regularization in tf
    Args:
        cost: Tensor containing the cost of the network without
              L2 regularization

    Returns: Tensor containing the cost of the network accounting
             for L2 regularization

    """

    return cost + tf.losses.get_regularization_losses()
