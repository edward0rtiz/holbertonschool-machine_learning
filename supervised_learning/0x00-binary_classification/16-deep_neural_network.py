##!/usr/bin/env python3
"""Script to create an DNN """

import numpy as np


class DeepNeuralNetwork():

    def __init__(self, nx, layers):

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("nodes must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if layers[i] < 0 or type(layers[i]) is not int:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                He_val = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.weights["W" + str(i + 1)] = He_val

