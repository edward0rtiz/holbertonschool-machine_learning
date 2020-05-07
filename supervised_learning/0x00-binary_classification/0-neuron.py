#!/usr/bin/env python3
"""Script to create a Neuron in a ANN"""

import numpy as np


class Neuron():
    """ Class Neuron """

    def __init__(self, nx):
        """

        Args:
            nx: Type int the number of n inputs into the ANN
        """
        if nx is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(nx).reshape(1, nx)  # Weight
        self.b = 0  # Bias
        self.A = 0  # output
