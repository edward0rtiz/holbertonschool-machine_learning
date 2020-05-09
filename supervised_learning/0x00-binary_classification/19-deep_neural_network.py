#!/usr/bin/env python3
"""Script to create cost
    method DNN
"""

import numpy as np


class DeepNeuralNetwork():
    """
    Class Deep Neural Network
    """

    def __init__(self, nx, layers):
        """

        Args:
            nx: input value
            nodes: nodes placed in the hidden layer
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for lay in range(self.L):
            if layers[lay] < 1 or type(layers[lay]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["b" + str(lay + 1)] = np.zeros((layers[lay], 1))
            if lay == 0:
                He_val = np.random.randn(layers[lay], nx) * np.sqrt(2 / nx)
                self.__weights["W" + str(lay + 1)] = He_val
            if lay > 0:
                He_val1 = np.random.randn(layers[lay], layers[lay - 1])
                He_val2 = np.sqrt(2 / layers[lay - 1])
                He_val3 = He_val1 * He_val2
                self.__weights["W" + str(lay + 1)] = He_val3

    @property
    def L(self):
        """
        Getter attr

        Returns: Private instance number of layers

        """
        return self.__L

    @property
    def cache(self):
        """
        Getter attr

        Returns: Private instance decit that hold intermediates
        values of the network

        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter attr
        Returns: Private instance holds weights and biases

        """
        return self.__weights

    def forward_prop(self, X):
        """
        Forward propagation function
        Args:
            X: x numpy array with shape (nx, m)

        Returns: forward propagation

        """
        self.__cache["A0"] = X
        for lay in range(self.__L):
            weights = self.__weights
            cache = self.__cache
            Za = np.matmul(weights["W" + str(lay + 1)], cache["A" + str(lay)])
            Z = Za + weights["b" + str(lay + 1)]
            cache["A" + str(lay + 1)] = 1 / (1 + np.exp(-Z))

        return cache["A" + str(self.__L)], cache

    def cost(self, Y, A):
        """
        Cost function using binary cross-entropy
        Args:
            Y: Y hat, slope
            A: Activated neuron output

        Returns: Cost value, efficiency when C = 0

        """

        m = Y.shape[1]
        C = (-1 / m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001 - A)))
        return C
