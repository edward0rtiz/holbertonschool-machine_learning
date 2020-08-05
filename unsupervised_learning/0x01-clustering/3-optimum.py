#!/usr/bin/env python3
"""Optimum K method aka inversed elbow"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Optimun K method with variance
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of
              clusters to check for (inclusive)
        kmax: positive integer containing the maximum number of
              clusters to check for (inclusive)
        iterations: positive integer containing the maximum number
                    of iterations for K-means
    Returns: results, d_vars, or None, None on failure
             results: list containing the outputs of K-means for
                     each cluster size
             d_vars: list containing the difference in variance from
                     the smallest cluster size for each cluster size
    """
    try:
        results = []
        d_vars = []
        minimun_k, _ = kmeans(X, kmin)
        variance_k = variance(X, minimun_k)
        for k in range(kmin, kmax + 1):
            cluster, clss = kmeans(X, k)
            results.append((cluster, clss))
            variance_d = variance(X, cluster)
            d_vars.append(variance_k - variance_d)
        return results, d_vars
    except Exception:
        return None, None
