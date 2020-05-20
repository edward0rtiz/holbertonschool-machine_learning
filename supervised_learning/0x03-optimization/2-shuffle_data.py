#!/usr/bin/env python3
"""Script to shuffle data in a matrix"""

import numpy as np


def shuffle_data(X, Y):
    """
    Function to shuffle data in a matrix
    Args:
        X: numpy.ndarray of shape (m, nx) to shuffle
        Y: numpy.ndarray of shape (m, ny) to shuffle
    Returns: the shuffled X and Y matrices

    """
    m = X.shape[0]
    shuffle = np.random.permutation(m)
    X_shuffled = X[shuffle]
    Y_shuffled = Y[shuffle]
    return X_shuffled, Y_shuffled
