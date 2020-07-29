#!/usr/bin/env python3
"""Shannon entropy and P affinities"""

import numpy as np


def HP(Di, beta):
    """
    Function that calculates shannon entropy
    Args:
        Di: numpy.ndarray of shape (n - 1,) containing the pairwise distances
            between a data point and all other points except itself
            n: number of data points
        beta: value for the Gaussian distribution

    Returns: (Hi, Pi)
             Hi: the Shannon entropy of the points
             Pi: a numpy.ndarray of shape (n - 1,) containing the P affinities
             of the points
    """
    P = np.exp(-Di * beta)
    sumP = np.sum(P)
    Pi = P / sumP
    Hi = -np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
