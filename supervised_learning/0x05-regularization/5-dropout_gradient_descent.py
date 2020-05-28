#!/usr/bin/env python3
"""Script to implement dropout in a gradient descent"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Function to use dropout in a gradient descent optimization model
    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) that contains
            the correct labels for the data
            classes: is the number of classes
            m: is the number of data points
        weights: dictionary of the weights and biases
                 of the neural network
        cache: dictionary of the outputs and dropout masks of each
               layer of the neural network
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network

    Returns:

    """
    m = Y.shape[1]
    W_copy = weights.copy()

    for i in reversed(range(L)):
        A = cache["A" + str(i + 1)]
        if i == L - 1:
            dZ = A - Y
            dW = (np.matmul(cache["A" + str(i)], dZ.T) / m).T
            db = np.sum(dZ, axis=1, keepdims=True) / m
        else:
            dW2 = np.matmul(W_copy["W" + str(i + 2)].T, dZ2)
            dtanh = 1 - (A * A)
            dZ = dW2 * dtanh
            dZ = dZ * cache["D" + str(i + 1)]
            dZ = dZ / keep_prob
            dW = np.matmul(dZ, cache["A" + str(i)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
        weights["W" + str(i + 1)] = (W_copy["W" + str(i + 1)] - (alpha * dW))
        weights["b" + str(i + 1)] = W_copy["b" + str(i + 1)] - (alpha * db)
        dZ2 = dZ
