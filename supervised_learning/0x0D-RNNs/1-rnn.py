#!/usr/bin/env python3
"""RNN Forward prop cell"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Forward prop on a simple RNN
    Args:
        rnn_cell: instance of RNNCell that will be used for the forward prop
        X: data to be used, given as a numpy.ndarray of shape (t, m, i)
           t: the maximum number of time steps
           m: the batch size
           i: the dimensionality of the data
        h_0:  initial hidden state, given as a numpy.ndarray of shape (m, h)
    Returns: H, Y
             H: numpy.ndarray containing all of the hidden states
             Y: numpy.ndarray containing all of the outputs
    """

    # list of caches that contains the hidden states
    Y = []

    # dimensions shape aka, time steps
    t, m, i = X.shape
    time_step = range(t)

    _, h = h_0.shape

    # initialize H with zeros
    H = np.zeros((t+1, m, h))
    H[0, :, :] = h_0

    # loop over time steps
    for ts in time_step:
        # Update next hidden state, compute the prediction
        h_next, y_pred = rnn_cell.forward(H[ts], X[ts])
        # Save the value of the new "next" hidden state
        H[ts+1, :, :] = h_next
        # Store values of the prediction
        Y.append(y_pred)

    Y = np.array(Y)
    return H, Y
