#!/usr/bin/env python3
""" Script to normalize a matrix"""


def normalize(X, m, s):
    """
    function to normalize a matrix
    Args:
        X: numpy.ndarray of shape (d, nx) to normalize
        m: numpy.ndarray of shape (nx,) that contains
            the mean of all features of X
        s: numpy.ndarray of shape (nx,) that contains
            the standard deviation of all features of X
    Returns: Normalized matrix

    """
    return (X - m) / s
