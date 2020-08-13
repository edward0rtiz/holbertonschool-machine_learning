#!/usr/bin/env python3
"""Steady state probabilities of a regular markov chain"""

import numpy as np


def regular(P):
    """
    Function that determines the steady state probabilities of a regular markov
    chain
    Args:
        P: 2D numpy.ndarray of shape (n, n) representing the transition matrix
           P[i, j]: is the probability of transitioning from state i to state j
           n: the number of states in the markov chain
    Returns: numpy.ndarray of shape (1, n) containing the steady state
             probabilities, or None on failure
    """
    try:
        if len(P.shape) != 2:
            return None
        n = P.shape[0]
        if n != P.shape[1]:
            return None

        # Method by eigendescomposition
        # Formula https://cutt.ly/Ed9Ad7s

        #  (πP).T = π.T ⟹ P.T π.T = π.T (.)
        evals, evecs = np.linalg.eig(P.T)

        # trick: has to be normalized
        state = (evecs / evecs.sum())

        # P.T π.T = π.T (.)
        new_state = np.dot(state.T, P)
        for i in new_state:
            if (i >= 0).all() and np.isclose(i.sum(), 1):
                return i.reshape(1, n)
    except Exception:
        return None
