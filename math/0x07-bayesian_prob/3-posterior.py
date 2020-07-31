#!/usr/bin/env python3
"""Posterior"""

import numpy as np
intersection = __import__('1-intersection').intersection
marginal = __import__('2-marginal').marginal


def posterior(x, n, P, Pr):
    """
    posterior function
    Args:
        x: the number of patients that develop severe side effects
        n: the total number of patients observed
        P: 1D numpy.ndarray containing the various hypothetical probabilities
           of developing severe side effects
        Pr: 1D numpy.ndarray containing the prior beliefs of P
    Returns: the the posterior probability of each probability in P given x
             and n, respectively
    """
    return intersection(x, n, P, Pr) / marginal(x, n, P, Pr)
