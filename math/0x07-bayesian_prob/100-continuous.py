#!/usr/bin/env python3
"""Coninuous posterior"""

from scipy import math, special
import numpy as np


def likelihood(x, n, P):
    return (special.factorial(n) / (special.factorial(x) *
                                    special.factorial(n - x))) * (P ** x) * \
           ((1 - P) ** (n - x))


def posterior(x, n, p1, p2):
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("{} must be a float in the range [0, 1]".format(p1))
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("{} must be a float in the range [0, 1]".format(p2))
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    like = likelihood(x, n, (p1 - p2))
    Pr = (p1 - p2) / 2
    intersection = like * Pr
    marginal = intersection + intersection
    pos = intersection / marginal
    return pos
