#!/usr/bin/env python3
"""Script of normalization constants in a matrix"""

import numpy as np


def normalization_constants(X):
    """
    function to get the stdev and mean
    Args:
        X: numpy.ndarray of shape (m, nx) to normalize
    Returns: mean and standard deviation of each feature, respectively

    """
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    return mean, stdev
