#!/usr/bin/env python3
"""Multivariate Normal Distribution"""

import numpy as np


class MultiNormal:
    def __init__(self, data):
        if type(data) is not np.ndarray:
            raise TypeError('data must be a 2D numpy.ndarray')
        if len(data.shape) != 2:
            raise ValueError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        n = data.shape[1]
        d = data.shape[0]
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        X_mean = data - self.mean
        self.cov = np.matmul(X_mean, X_mean.T) / (n - 1)
