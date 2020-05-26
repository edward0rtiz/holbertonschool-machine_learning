#!/usr/bin/env python3
"""Script to update wieghts and biases of a DNN
    with gradient descent and L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function to implement L2 regulatization using gradient descent
    Args:
        Y: one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels for the data
            classes: is the number of classes
            m: is the number of data points
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs of each layer of the neural network
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers of the network

    Returns: Cost of the network accounting for L2 regularization

    """
    m = Y.shape[1]
    weights_copy = weights.copy()

    for i in reversed(range(L)):
        A = cache["A" + str(i + 1)]
        if i == L - 1:
            dZ = cache["A" + str(i + 1)] - Y
            dW = (np.matmul(cache["A" + str(i)], dZ.T) / m).T
            dW_L2 = dW + (lambtha / m) * weights_copy["W" + str(i + 1)]
            db = np.sum(dZ, axis=1, keepdims=True) / m
        else:
            dW2 = np.matmul(weights_copy["W" + str(i + 2)].T, dZ2)
            tanh = np.exp(A)
            dZ = dW2 * tanh
            dW = np.matmul(dZ, cache["A" + str(i)].T) / m
            dW_L2 = dW + (lambtha / m) * weights_copy["W" + str(i + 1)]
            db = np.sum(dZ, axis=1, keepdims=True) / m
        weights["W" + str(i + 1)] = (weights_copy["W" + str(i + 1)] - (alpha * dW_L2))
        weights["b" + str(i + 1)] = weights_copy["b" + str(i + 1)] - (alpha * db)
        dZ2 = dZ