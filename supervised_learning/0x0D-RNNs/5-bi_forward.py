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
