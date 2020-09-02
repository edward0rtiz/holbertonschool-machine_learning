#!/usr/bin/env python3
"""GRU cell for a RRN"""

import numpy as np


class GRUCell:
    """ Gated recurrent unit """

    def __init__(self, i, h, o):
        """
        Initializer constructor
        Args:
            i: the dimensionality of the data
            h: the dimensionality of the hidden state
            o: he dimensionality of the outputs
        """

        # weight for the cell
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        # Bias of the cell
        self.bz = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """softmax function"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """
        forward propagation for one time step
        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                    m: hidden state
            x_t: numpy.ndarray of shape (m, i) that contains the data
                 input for the cell
        Returns: h_next, y
                 h_next: the next hidden state
                 y: the output of the cell
        """
        # previous hidden cell state
        h_x = np.concatenate((h_prev.T, x_t.T), axis=0)

        # update gate z vector (TYPE 1)
        zt = self.sigmoid((h_x.T @ self.Wz) + self.bz)

        # reset gate vector (TYPE 1)
        rt = self.sigmoid((h_x.T @ self.Wr) + self.br)

        # cell operation after updated z
        h_x = np.concatenate(((rt * h_prev).T, x_t.T), axis=0)

        # x_t activated via tanh to get candidate activation vector
        ht_c = np.tanh((h_x.T @ self.Wh) + self.bh)

        # compute output vector
        h_next = (1 - zt) * h_prev + zt * ht_c

        # final output of the cell
        y = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, y
