#!/usr/bin/env python3
"""Multivariate Normal Distribution"""

import numpy as np


class MultiNormal():
    """
    Multinormal class
    """
    def __init__(self, data):
        """
        Init method
        Args:
            data: numpy.ndarray of shape (d, n) of the data set
        """
        if type(data) is not np.ndarray:
            raise TypeError('data must be a 2D numpy.ndarray')
        if len(data.shape) != 2:
            raise ValueError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        d, n = data.shape
        self.mean = np.mean(data, axis=1, keepdims=True)
        X_mean = data - self.mean
        self.cov = np.dot(X_mean, X_mean.T) / (n - 1)
