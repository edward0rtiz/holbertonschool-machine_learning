#!/usr/bin/env python3
"""Posterior"""

import numpy as np
from scipy import math, special


def intersection(x, n, P, Pr):
    numerator = np.math.factorial(n)
    denominator = (np.math.factorial(x) * (np.math.factorial(n - x)))
    factorial = numerator / denominator
    P_likelihood = factorial * (np.power(P, x)) * (np.power((1 - P), (n - x)))
    intersection = P_likelihood * Pr
    return intersection


def marginal(x, n, P, Pr):
    inter = intersection(x, n, P, Pr)
    return np.sum(inter)


def posterior(x, n, p1, p2):
    """
    Continuous
    Args:
        x: the number of patients that develop severe side effects
        n: the total number of patients observed
        p1: the lower bound on the range
        p2: the upper bound on the range
    Returns: the posterior probability that p is within the range
             [p1, p2] given x and n

    """
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError('All values in p1 must be in the range [0, 1]')
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError('All values in p2 must be in the range [0, 1]')
    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')
    P = (p2 - p1) / 2
    Pr = p2 - p1
    pos = intersection(x, n, P, Pr) / marginal(x, n, P, Pr)
    return pos
