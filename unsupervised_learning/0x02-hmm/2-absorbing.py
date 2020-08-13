#!/usr/bin/env python3
""" Absorbing Markov chain"""

import numpy as np


def absorbing(P):
    """
    Function that determines if a markov chain is absorbing
    Args:
        P: 2D numpy.ndarray of shape (n, n) representing the transition matrix
           P[i, j]: is the probability of transitioning from state i to state j
           n: the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    if len(P.shape) != 2:
        return None
    n1, n2 = P.shape
    if (n1 != n2) or type(P) is not np.ndarray:
        return None
    D = np.diagonal(P)
    if (D == 1).all():
        return True
    if not (D == 1).any():
        return False

    # formula t=(I-Q)^-1 1
    # approach using Neumman series: t = (∑ inf k=0 * Q^k) * 1
    # suggested implementation based on: https://stackoverflow.com/
    # questions/45164505/best-iterative-way-to-calculate-the-fundamental-matrix-
    # of-an-absorbing-markov-ch
    for i in range(n1):
        # print('this is Pi {}'.format(P[i]))
        for j in range(n2):
            # print('this is Pj {}'.format(P[j]))
            if (i == j) and (i + 1 < len(P)):
                if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                    return False
    return True
