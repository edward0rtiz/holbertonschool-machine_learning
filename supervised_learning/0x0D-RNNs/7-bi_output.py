#!/usr/bin/env python3
"""Bi-directional Cell of a RNN"""

import numpy as np


class BidirectionalCell:
    """Bi-directional cell class"""

    def __init__(self, i, h, o):
        """
        Initializer constructor
        Args:
            i: the dimensionality of the data
            h: the dimensionality of the hidden state
            o: he dimensionality of the outputs
        """

        # weight for the cell
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))

        # Bias of the cell
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward prop for a bidirectional cell
        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                    hidden state
            x_t: numpy.ndarray of shape (m, i) that contains the data
                 input for the cell
        Returns: h_next, the next hidden state
        """
        # previous hidden cell state
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)

        # compute output vector
        h_next = np.tanh((h_x.T @ self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Backwardprop for a bidirectional cell
        Args:
            h_next: numpy.ndarray of shape (m, h) containing the next hidden
                    state
            x_t: numpy.ndarray of shape (m, i) that contains the data input for
                 the cell
        Returns:
        """

        # previous hidden cell state
        h_x = np.concatenate((h_next.T, x_t.T), axis=0)

        # compute output vector
        h_pev = np.tanh((h_x.T @ self.Whb) + self.bhb)

        return h_pev

    def softmax(self, x):
        """softmax function"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def output(self, H):
        """
        Calculate outputs of the RNN
        Args:
            H: numpy.ndarray of shape (t, m, 2 * h) that contains the
               concatenated hidden states from both directions, excluding their
               initialized states
               t: the number of time steps
               m: the batch size for the data
               h: the dimensionality of the hidden states
        Returns: Y being the outputs
        """
        # dimensions shape aka, time steps
        t, m, _ = H.shape
        time_step = range(t)

        # Bias of presynaptic node
        o = self.by.shape[1]
        print("#$$##$#$")
        print(o)

        Y = np.zeros((t, m, o))

        for ts in time_step:
            # final output of the cell
            y_pred = self.softmax((H[ts] @ self.Wy) + self.by)
            Y[ts] = y_pred
        return Y
