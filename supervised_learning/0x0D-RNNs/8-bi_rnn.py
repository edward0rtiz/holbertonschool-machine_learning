#!/usr/bin/env python3
"""Bi-directional for a RNN"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Forward prop for a deep RNN
    Args:
        bi_cell: instance of BidirectinalCell that will be used for the forward
                 propagation
        X: data to be used, given as a numpy.ndarray of shape (t, m, i)
           t: the maximum number of time steps
           m: the batch size
           i: the dimensionality of the data
        h_0: initial hidden state in the forward direction, given as a
             numpy.ndarray of shape (m, h)
             h: the dimensionality of the hidden state
        h_t: initial hidden state in the backward direction, given as a
             numpy.ndarray of shape (m, h)
    Returns: H, Y
             H: numpy.ndarray containing all of the concatenated hidden states
             Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    time_step = range(t)

    _, h = h_0.shape

    H_f = np.zeros((t+1, m, h))
    H_b = np.zeros((t+1, m, h))

    # Initialization
    H_f[0] = h_0
    H_b[t] = h_t

    for ti in time_step:
        H_f[ti+1] = bi_cell.forward(H_f[ti], X[ti])

    # Reversed iteration
    for ri in range(t-1, -1, -1):
        H_b[ri] = bi_cell.backward(H_b[ri+1], X[ri])
    H = np.concatenate((H_f[1:], H_b[:t]), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
