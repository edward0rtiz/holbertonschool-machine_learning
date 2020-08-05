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

    # formula
    # p(x∣ μ,Σ) = (1 √(2π)d|Σ|)exp(−1/2(x−μ)T Σ−1(x−μ))
    n, d = X.shape
    mean = m
    x_m = X - mean
    x_mT = x_m.T

    # Determinant of the covariance matrix (d x d)
    det_S = np.linalg.det(S)

    # Since Σ is Hermitian, it has an eigendecomposition
    inv_S = np.linalg.inv(S)

    # Formula Section one: (1 √(2π)d|Σ|)
    part_1 = (1 / np.sqrt((2 * np.pi) ** d * det_S))
    print(part_1)
    part_2 = - 0.5 * (np.dot(x_m, inv_S).T.dot(x_m))

    pdf = part_1 * np.exp(part_2)

    P = np.where(pdf < 1e-300, 1e-300, pdf)
    return P
