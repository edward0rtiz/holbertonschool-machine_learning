#!/usr/bin/env python3
"""Correlation function"""

import numpy as np


def correlation(C):
    """
    Function that calculates a correlation matrix
    Args:
        C: numpy.ndarray of shape (d, d) containing a covariance matrix
    Returns: numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2:
        raise ValueError('C must be a 2D square matrix')
    if C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    # corr(X) = (diag(Kxx)) -1/2 Kxx (diag(Kxx)) -1/2from
    cov = np.diag(C)
    std_x = np.sqrt(cov)
    std_y = std_x
    std_product = std_x * std_y
    corr = C / std_product

    return corr
