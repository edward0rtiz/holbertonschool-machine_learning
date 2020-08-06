#!/usr/bin/env python3
""" M-step: re-estimate all parameters so that
the likelihood of observing what we observed is maximized
"""

import numpy as np


def maximization(X, g):
    """
    calculates the maximization step in the EM algorithm for a GMM
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        g: numpy.ndarray of shape (k, n) containing the posterior
           probabilities for each data point in each cluster
    Returns: pi, m, S, or None, None, None on failure
             pi: numpy.ndarray of shape (k,) containing the updated priors
                 for each cluster
             m: numpy.ndarray of shape (k, d) containing the updated centroid
                means for each cluster
             S: numpy.ndarray of shape (k, d, d) containing the updated
                covariance matrices for each cluster
    """

    gaussian_components = g

    n, d = X.shape
    k = gaussian_components.shape[0]

    # pi
    priors = np.zeros((k,))

    # m
    centroid_updated = np.zeros((k, d))

    # S
    covariance_updated = np.zeros((k, d, d))

    # Formula https://bit.ly/31pkdox
    for i in range(k):

        # Mu components
        # Needed to adjust dimensions for fitting the covariance
        mu_up = np.sum((gaussian_components[i, :, np.newaxis] * X), axis=0)
        mu_down = np.sum(gaussian_components[i], axis=0)
        centroid_updated[i] = mu_up / mu_down

        # Sigma components
        x_m = X - centroid_updated[i]
        sigma_up = np.dot(gaussian_components[i] * x_m.T, x_m)
        sigma_down = np.sum(gaussian_components[i])
        covariance_updated[i] = sigma_up / sigma_down

        # Pi =  priors after computing derivation of sigma and mu
        # Formula: P(j) = n(j) / n = Σn i=1 P(j|i) / n
        priors[i] = np.sum(gaussian_components[i]) / n
    return priors, centroid_updated, covariance_updated
