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
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        d, n = data.shape
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        X_mean = data - self.mean
        self.cov = np.matmul(X_mean, X_mean.T) / (n - 1)

    def pdf(self, x):
        """
        PDF function
        Args:
            X: numpy.ndarray of shape (d, 1) containing the data point
               whose PDF should be calculated

        Returns:
        """
        if type(x) is not np.ndarray:
            raise TypeError('x must be a numpy.ndarray')
        if len(x.shape) != 2:
            raise ValueError('x must have the shape ({}, 1)'.
                             format(self.cov.shape[0]))
        if x.shape[1] != 1:
            raise ValueError('x must have the shape ({}, 1)'.
                             format(self.cov.shape[0]))
        if x.shape[0] != self.cov.shape[0]:
            raise ValueError('x must have the shape ({}, 1)'.
                             format(self.cov.shape[0]))

        X_mean = x - self.mean

        d = self.cov.shape[0]
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        p_a = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        p_b1 = np.dot(-(X_mean).T, inv)
        p_b2 = np.dot(p_b1, (X_mean) / 2)
        p_c = np.exp(p_b2)
        pdf = float(p_a * p_c)
        return pdf
