#!/usr/bin/env python3
"""Script to implement forward propagation in a FNN
    with private attributes
"""

import numpy as np


class NeuralNetwork():
    """ Class neural network"""

    def __init__(self, nx, nodes):
        """

        Args:
            nx: input value
            nodes: nodes placed in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.randn(nodes).reshape(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Weight vector 1 hidden layer

        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Bias1

        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Activated1

        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Weight vector 2

        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Bias2

        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter attr
        Args:
            self: Private attribute

            Returns: Activated output 2 prediction

        """
        return self.__A2

    def forward_prop(self, X):
        """
        Method to calculate a forward propagation in a FNN
        Args:
            X: input data vector

        Returns: A1 and A2 the activation nodes using sigmoid

        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        sigmoid_1 = 1 / (1 + np.exp(-Z1))
        self.__A1 = sigmoid_1
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        sigmoid_2 = 1 / (1 + np.exp(-Z2))
        self.__A2 = sigmoid_2

        return self.__A1, self.__A2
