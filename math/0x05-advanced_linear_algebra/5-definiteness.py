#!/usr/bin/env python3
"""Definiteness functions"""

import numpy as np
from numpy import linalg as LA


def definiteness(matrix):
    """
    Definiteness function
    Args:
        matrix: numpy.ndarray of shape (n, n)
                whose definiteness should be calculated
    Returns: positive definite, positive semi-definite,
             negative semi-definite, negative definite of
             indefinite, respectively
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) == 1:
        return None
    if (matrix.shape[0] != matrix.shape[1]):
        return None

    w, v = LA.eig(matrix)
    if np.all(w > 0):
        return "Positive definite"
    if np.all(w >= 0):
        return "Positive semi-definite"
    if np.all(w < 0):
        return "Negative definite"
    if np.all(w <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
