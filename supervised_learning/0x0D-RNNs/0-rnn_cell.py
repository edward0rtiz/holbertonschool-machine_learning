#!/usr/bin/env python3
"""Class RNNCell"""

import numpy as np


class RNNCell:
    """Vanilla model fro a RNN cell"""

    def __init__(self, i, h, o):
        """
        Initialize class constructor
        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """

        # Wh instance contains Whh= (h, h) + Whx(i, h)
        self.Wh = np.random.normal(size=(i + h, h))
        # Wy instance
        self.Wy = np.random.normal(size=(h, o))

        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        forward propagation vanilla RNN cell
        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                    hidden state
            x_t: numpy.ndarray of shape (m, i) that contains the data input
                 for the cell
                 m: batche size for the data
        Returns: h_next, y
                 h_next is the next hidden state
                 y is th output of the cell
        """

        # previous hidden cell state
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)

        # compute next activation state
        h_next = np.tanh((h_x.T @ self.Wh) + self.bh)

        # # compute output of the current cell using the formula given above
        y_pred = (h_next @ self.Wy) + self.by  # z (sum of weight and X's)
        z = y_pred  # prediction at timestep

        # (σ): g(z) = e^z / (1 + e^{-z})
        y = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

        return h_next, y
