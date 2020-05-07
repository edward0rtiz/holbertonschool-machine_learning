#!/usr/bin/env python3
"""Script to forward propagate a neuron"""

import numpy as np


class Neuron():
    """ Class Neuron """

    def __init__(self, nx):
        """

        Args:
            nx: Type int the number of n inputs into the ANN
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(nx).reshape(1, nx)  # Weight
        self.__b = 0  # Bias
        self.__A = 0  # output

    @property
    def W(self):
        """

        Returns: private instance weight

        """
        return self.__W

    @property
    def b(self):
        """

        Returns: private instance bias

        """
        return self.__b

    @property
    def A(self):
        """

        Returns: private instance output

        """
        return self.__A

    def forward_prop(self, X):
        """
        Function of forward propagation
        activated by a sigmoid function

        Args:
            X: ndarray with shape of nx, m

        Returns: neuron activated using sigmoid

        """
        EWx = np.matmul(self.__W, X) + self.__b  # z (sum of weight and X's)
        sigmoid = 1 / (1 + np.exp(-EWx))  # (σ): g(z) = 1 / (1 + e^{-z})
        self.__A = sigmoid
        return self.__A
