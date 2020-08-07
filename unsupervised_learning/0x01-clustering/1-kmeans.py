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
    """
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
    """
    n, d = X.shape

    min_val = np.amin(X, axis=0)
    max_val = np.amax(X, axis=0)

    C = np.random.uniform(min_val, max_val, (k, d))
    C_prev = np.copy(C)

    X_ = X[:, :, np.newaxis]
    C_ = C.T[np.newaxis, :, :]
    diff = X_ - C_
    D = np.linalg.norm(diff, axis=1)

    clss = np.argmin(D, axis=1)

    for i in range(iterations):

        for j in range(k):
            # recalculate centroids
            index = np.where(clss == j)
            if len(index[0]) == 0:
                C[j] = np.random.uniform(min_val, max_val, (1, d))
            else:
                C[j] = np.mean(X[index], axis=0)

        X_ = X[:, :, np.newaxis]
        C_ = C.T[np.newaxis, :, :]
        diff = X_ - C_
        D = np.linalg.norm(diff, axis=1)

        # Eucledean norm (alternative)
        # (a - b)**2 = a^2 - 2ab + b^2 expansion
        """
        a2 = np.sum(C ** 2, axis=1)[:, np.newaxis]
        b2 = np.sum(X ** 2, axis=1)
        ab = np.matmul(C, X.T)
        D = np.sqrt(a2 - 2 * ab + b2)
        """

        clss = np.argmin(D, axis=1)

        if (C == C_prev).all():
            return C, clss
        C_prev = np.copy(C)

    return C, clss
