#!/usr/bin/env python3
"""Markov chain probability"""


import numpy as np


def markov_chain(P, s, t=1):
    """
    Function that deteermines the probability of a markov chain in a particular
    state after a specified number of iterations
    Args:
        P: 2D numpy.ndarray of shape (n, n) representing the transition matrix
           P[i, j]: is the probability of transitioning from state i to state j
           n: the number of states in the markov chain
        s: numpy.ndarray of shape (1, n) represents the probability of starting
           in each state
        t: number of iterations that the markov chain has been through
    Returns: numpy.ndarray of shape (1, n)
    """
    try:

        if len(P.shape) != 2:
            return None
        n1, n2 = P.shape
        if (n1 != n2) or type(P) is not np.ndarray or not isinstance(t, int):
            return None
        if t < 0:
            return None
        if n1 != s.shape[1] or s.shape[0] != 1:
            return None
        for i in range(t):
            s = np.matmul(s, P)
        return s
    except Exception:
        return None

