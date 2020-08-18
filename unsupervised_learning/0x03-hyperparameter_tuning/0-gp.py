#!/usr/bin/env python3
"""
Gaussian Process
"""

import numpy as np


class GaussianProcess():
    """
    Class constructor that represents a noiseless 1D GP
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Init method
        Args:
            X_init: numpy.ndarray of shape (t, 1) representing the inputs
                    already sampled with the black-box function
                    t: the number of initial samples
            Y_init: numpy.ndarray of shape (t, 1) representing the outputs
                    of the black-box function for each input in X_init
                    t: the number of initial samples
            l: the length parameter for the kernel
            sigma_f: standard deviation given to the output of the black-box
                     function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        kernel function aka(covariance function)
        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)
        Returns: covariance kernel matrix as a numpy.ndarray of shape (m, n)
        """

        # formula κ(xi,xj)=σ^2f exp(−12l2(xi−xj)T(xi−xj))(6)
        # source: http://krasserm.github.io/2018/03/19/gaussian-processes/
        sqdist1 = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)
        sqdist2 = 2 * np.dot(X1, X2.T)
        sqdist = sqdist1 - sqdist2
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
