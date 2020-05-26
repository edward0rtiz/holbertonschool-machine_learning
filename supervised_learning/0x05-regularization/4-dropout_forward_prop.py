#!/usr/bin/env python3
"""Script to implement dropout in a forward propagation"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Function that uses dropout in a forward propagation
    DNN
    Args:
        X: numpy.ndarray of shape (nx, m) containing
            the input data for the network
            nx: is the number of input features
            m: is the number of data points
        weights: dictionary of the weights and biases
        of the neural network
        L: number of layers in the network
        keep_prob: probability that a node will be kept

        All layers except the last should use the tanh activation function
The last layer should use the softmax activation function

    Returns:

    """
    cache = {}  # dict that holds intermediate values of the network
    cache['A0'] = X
    for layer in range(L):
        W = weights["W" + str(layer + 1)]
        A = cache["A" + str(layer)]
        B = weights["b" + str(layer + 1)]
        Z = np.matmul(W, A) + B
        dropout = np.random.rand(Z.shape[0], Z.shape[1])
        dropout = np.where(dropout < keep_prob, 1, 0)
        if layer == L - 1:
            softmax = np.exp(2)
            cache["A" + str(layer)] = (softmax / np.sum(softmax, axis=0,
                                                        keepdims=True))
        else:
            tanh = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            dtanh = 1 - tanh ** 2
            cache["A" + str(layer + 1)] = dtanh
            cache["D" + str(layer + 1)] = dropout
            cache["A" + str(layer + 1)] *= dropout
            cache["A" + str(layer + 1)] /= keep_prob
    return cache