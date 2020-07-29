#!/usr/bin/env python3
"""computing grads"""

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculate the gradients of Y
    Args:
        Y: numpy.ndarray of shape (n, ndim) containing the low
           dimensional transformation of X
        P: numpy.ndarray of shape (n, n) containing the P
           affinities of X
    Returns: (dY, Q)
             dY: dY is a numpy.ndarray of shape (n, ndim) containing the
             gradients of Y
             Q: Q is a numpy.ndarray of shape (n, n) containing the Q
                affinities of Y
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    dY = np.zeros((n, ndim))

    # for i in range(n):
    #    dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i],
    #    (ndim, 1)).T * (Y[i, :] - Y), 0)
    # return (dY, Q)
    PQ = P - Q
    PQ_expanded = np.expand_dims((PQ * num).T, axis=2)

    for i in range(n):
        y_diff = Y[i, :] - Y
        # dY[i, :] = 4. * (PQ_expanded * y_diff).sum(1)
        dY[i, :] = np.sum((PQ_expanded[i, :] * y_diff), 0)

    return dY, Q
