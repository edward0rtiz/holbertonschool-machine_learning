#!/usr/bin/env python3
"""LSTM cell for a RRN"""

import numpy as np


class LSTMCell:
    """ LSTMCell unit """

    def __init__(self, i, h, o):
        """
        Initializer constructor
        Args:
            i: the dimensionality of the data
            h: the dimensionality of the hidden state
            o: he dimensionality of the outputs
        """

        # weight for the cell
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        # Bias of the cell
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """softmax function"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, c_prev, x_t):
        """
        forward propagation for one time step in a LSTM
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

        # forget gate activation vector
        ft = self.sigmoid((h_x.T @ self.Wf) + self.bf)

        # input/update gate activation vector
        it = self.sigmoid((h_x.T @ self.Wu) + self.bu)

        # candidate value
        cct = np.tanh((h_x.T @ self.Wc) + self.bc)
        c_next = ft * c_prev + it * cct

        # output gate
        ot = self.sigmoid((h_x.T @ self.Wo) + self.bo)

        # compute hidden state
        h_next = ot * np.tanh(c_next)

        # final output of the cell
        y = self.softmax((h_next @ self.Wy) + self.by)

        return h_next, c_next, y
