#!/usr/bin/env python3
"""PDF function """

import numpy as np


def pdf(X, m, S):
    """
    Probability Density Function of gaussian distributions
    Args:
        X: numpy.ndarray of shape (n, d) containing the data points whose
           PDF should be evaluated
        m: numpy.ndarray of shape (d,) containing the mean of the distribution
        S: numpy.ndarray of shape (d, d) containing the covariance of the
           distribution
    Returns: P, or None on failure
             P: numpy.ndarray of shape (n,) containing the PDF values for each
                data point.
                All values in P should have a minimum value of 1e-300
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    # formula
    # p(x∣ μ,Σ) = (1 √(2π)d|Σ|)exp(−1/2(x−μ)T Σ−1(x−μ))
    n, d = X.shape
    mean = m
    x_m = X - mean

    # Determinant of the covariance matrix (d x d)
    det_S = np.linalg.det(S)

    # Since Σ is Hermitian, it has an eigendecomposition
    inv_S = np.linalg.inv(S)

    # Formula Section one: (1 √(2π)d|Σ|)
    part_1_dem = np.sqrt((2 * np.pi) ** d * det_S)

    # Formula Section two_upper_1: −1/2(x−μ)T
    part_2 = np.matmul(x_m, inv_S)

    # Formula Section two_upper_2: Σ−1(x−μ) used diagonal to fix alloc err
    part_2_1 = np.sum(x_m * part_2, axis=1)

    # Formula Section two exp(−1/2(x−μ)T Σ−1(x−μ))
    part_2_2 = np.exp(part_2_1 / -2)

    # pdf = part_1 * part_2_2:
    pdf = part_2_2 / part_1_dem
    P = np.where(pdf < 1e-300, 1e-300, pdf)
    return P
