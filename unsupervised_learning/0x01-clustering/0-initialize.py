#!/usr/bin/env python3
"""Initialize function"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
           n: the number of data points
           d: the number of dimensions for each data point
        k: positive integer containing the number of clusters

    Returns: numpy.ndarray of shape (k, d) containing the initialized
             centroids for each cluster, or None on failure
    """
    n, d = X.shape
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    return np.random.uniform(X_min, X_max, size=(k, d))

