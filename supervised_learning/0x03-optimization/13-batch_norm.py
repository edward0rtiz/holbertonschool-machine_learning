#!/usr/bin/env python3
"""Script to normalize a batch for a DNN"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Function to normalize a batch
    Args:
        Z: a numpy.ndarray of shape (m, n) that should be normalized
            m is the number of data points
            n is the number of features in Z
        gamma: numpy.ndarray of shape (1, n) containing the scales used
                for batch normalization
        beta: numpy.ndarray of shape (1, n) containing the offsets used
                for batch normalization
        epsilon: small number used to avoid division by zero

    Returns: the normalized Z matrix

    """
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    std = np.sqrt(variance + epsilon)
    Z_centered = Z - mean
    Z_normalization = Z_centered / std
    Z_batch_norm = gamma * Z_normalization + beta

    return Z_batch_norm
