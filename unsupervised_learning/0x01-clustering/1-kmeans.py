#!/usr/bin/env python3
"""K means"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    K-means on a data set
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
           n: the number of data points
           d: the number of dimensions for each data point
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of
        iterations that should be performed
    Returns: C, clss, or None, None on failure
             C: numpy.ndarray of shape (k, d) containing the centroid means
                for each cluster
             clss: numpy.ndarray of shape (n,) containing the index of
                  the cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Setting min and max values per col
    n, d = X.shape
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # Centroid
    C = np.random.uniform(X_min, X_max, size=(k, d))

    # Loop for the maximum number of iterations
    for i in range(iterations):

        # initializes k centroids by selecting them from the data points
        centroids = np.copy(C)
        centroids_extended = C[:, np.newaxis]

        # distances also know as euclidean distance
        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
        # an array containing the index to the nearest centroid for each point
        clss = np.argmin(distances, axis=0)

        # Assign all points to the nearest centroid
        for c in range(k):
            if X[clss == c].size == 0:
                C[c] = np.random.uniform(X_min, X_max, size=(1, d))
            else:
                C[c] = X[clss == c].mean(axis=0)

        centroids_extended = C[:, np.newaxis]
        distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
        clss = np.argmin(distances, axis=0)

        if (centroids == C).all():
            break

    return C, clss
