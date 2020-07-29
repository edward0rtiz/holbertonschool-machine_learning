#!/usr/bin/env python3
"""Cost function"""

import numpy as np


def cost(P, Q):
    """
    Cost function
    Args:
        P: numpy.ndarray of shape (n, n) containing the P affinities
        Q: numpy.ndarray of shape (n, n) containing the Q affinities
    Returns: C, the cost of the transformation
    """
    # C = Smi,Smj (Pj|i log(Pj|i / Qj|i))
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    C = np.sum(P * np.log(P / Q))
    return C
