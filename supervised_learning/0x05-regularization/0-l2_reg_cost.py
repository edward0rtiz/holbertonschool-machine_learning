#!/usr/bin/env python3
"""Script to use L2 regularization in a DNN"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Fucntion to implement L2 regularization
    Args:
        cost: cost of the network without L2 regularization
        lambtha: the regularization parameter
        weights: dictionary of the weights and biases
                (numpy.ndarrays) of the neural network
        L: number of layers in the neural network
        m: number of data points used

    Returns: cost of the network accounting for L2 regularization

    """
    norm = 0
    for key, values in weights.items():
        if key[0] == 'W':
            norm = norm + np.linalg.norm(values)
    return cost + (lambtha / (2 * m) * norm)
