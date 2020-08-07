#!/usr/bin/env python3
"""K-means using scikit learn"""

import sklearn.cluster


def kmeans(X, k):
    """
    performs K-means on a dataset
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters
    Returns: C, clss
             C: numpy.ndarray of shape (k, d) containing the centroid
                means for each cluster
             clss: numpy.ndarray of shape (n,) containing the index of
                   the cluster in C that each data point belongs to
    """
    k_mean = sklearn.cluster.KMeans(n_clusters=k)
    k_mean.fit(X)
    clss = k_mean.labels_
    C = k_mean.cluster_centers_

    return C, clss
