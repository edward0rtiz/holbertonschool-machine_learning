#!/usr/bin/env python3
"""GMM"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    initializes variables for a Gaussian Mixture Model
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
    Returns: pi, m, S, or None, None, None on failure
            pi: numpy.ndarray of shape (k,) containing the priors
                for each cluster initialized evenly
            m: numpy.ndarray of shape (k, d) containing the centroid
                means for each cluster, initialized with K-means
            s: numpy.ndarray of shape (k, d, d) containing the covariance
               matrices for each cluster, initialized as identity matrices
    """
    n, d = X.shape

    # priors for each cluster, initialized evenly
    phi = np.ones(k)/k

    # centroid means for each cluster, initialized with K-means
    m, _ = kmeans(X, k)

    #  covariance matrices for each cluster, initialized as identity matrices
    S = np.tile(np.identity(d), (k, 1)).reshape(k, d, d)

    return phi, m, S
