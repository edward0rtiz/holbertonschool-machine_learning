#!/usr/bin/env python3
"""Coninuous posterior"""

from scipy import special
import numpy as np


def posterior(x, n, p1, p2):
    """
    continuous posterior
    Args:
        x: the number of patients that develop severe side effects
        n:  the total number of patients observed
        p1: he lower bound on the range
        p2: the upper bound on the range
    Returns: posterior probability that p is within the range
            [p1, p2] given x and n
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(p1) is not float or not 0 <= p1 <= 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if type(p2) is not float or not 0 <= p2 <= 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    def likelihood(x, n, P):
        return (special.factorial(n) / (special.factorial(x) *
                                        special.factorial(n - x))) \
               * (P ** x) * ((1 - P) ** (n - x))

    P = (x - (p1 + p2)) / (n - (p1 + p2))
    like = likelihood(x, n, P)
    Pr = p2 - p1
    intersection = like * Pr
    marginal = np.sum(intersection)
    pos = intersection / marginal

    return pos
