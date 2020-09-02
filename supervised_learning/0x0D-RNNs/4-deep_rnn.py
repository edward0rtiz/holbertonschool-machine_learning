#!/usr/bin/env python3
"""forward propagation to a deep RNN"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Forward prop in a deep RNN
    Args:
        rnn_cells: list of RNNCell instances of length l that will be used
                   for the forward propagation
                   l: number of layers
        X: data to be used, given as a numpy.ndarray of shape (t, m, i)
           t: maximum number of time steps
           m: batch size
           i: dimensionality of the data
        h_0: initial hidden state, given as a numpy.ndarray of shape (l, m, h)
             h: h is the dimensionality of the hidden state
    Returns: H, Y
             H: numpy.ndarray containing all of the hidden states
             Y: numpy.ndarray containing all of the outputs
    """

    # list of caches that contains the hidden states
    Y = []

    # dimensions shape aka, time steps
    t, m, i = X.shape
    _, _, h = h_0.shape

    time_step = range(t)
    layers = len(rnn_cells)

    # initialize H with zeros
    H = np.zeros((t+1, layers, m, h))
    H[0, :, :, :] = h_0

    # loop over time steps
    for ts in time_step:
        for ly in range(layers):
            if ly == 0:
                # Update next hidden state, compute the prediction
                h_next, y_pred = rnn_cells[ly].forward(H[ts, ly], X[ts])
            else:
                # Update next hidden state, compute the prediction
                h_next, y_pred = rnn_cells[ly].forward(H[ts, ly], h_next)
            # Save the value of the new "next" hidden state
            H[ts+1, ly, :, :] = h_next
        # Store values of the prediction
        Y.append(y_pred)
    Y = np.array(Y)
    return H, Y
