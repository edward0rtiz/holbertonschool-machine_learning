#!/usr/bin/env python3
"""Intersection"""

import numpy as np


def intersection(x, n, P, Pr):
    """
    Intersection function
    Args:
        x: the number of patients that develop severe side effects
        n: the total number of patients observed
        P: 1D numpy.ndarray containing the various hypothetical probabilities
           of developing severe side effects
        Pr: 1D numpy.ndarray containing the prior beliefs of P
    Returns: 1D numpy.ndarray containing the intersection of obtaining x and n
             with each probability in P, respectively
    """
    if not isinstance(n, int) or (n <= 0):
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or (x < 0):
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')
    #if not isinstance(Pr, np.ndarray) or ()

    numerator = np.math.factorial(n)
    denominator = (np.math.factorial(x) * (np.math.factorial(n - x)))
    factorial = numerator / denominator
    P_likelihood = factorial * (np.power(P, x)) * (np.power((1 - P), (n - x)))
    intersection = P_likelihood * Pr
    return intersection
