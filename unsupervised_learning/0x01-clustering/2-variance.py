#!/usr/bin/env python3
"""Variance"""

import numpy as np


def variance(X, C):
    """
    Calculate the total intra-cluste variance for a data set
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        C: numpy.ndarray of shape (k, d) containing the centroid means
           for each cluster
    Returns: Var or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape != C.shape:
        return None

    n, d = X.shape

    # distances also know as euclidean distance
    centroids_extended = C[:, np.newaxis]
    distances = np.sqrt(((X - centroids_extended) ** 2).sum(axis=2))
    min_distances = np.min(distances, axis=0)

    # W=∑k=1K∑xi∈Ck∥xi−x¯k∥2
    variance = np.sum(min_distances ** 2)

    return variance
