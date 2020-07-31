#!/usr/bin/env python3
"""Coninuous posterior"""

from scipy import math, special


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
    return 0.6098093274896035