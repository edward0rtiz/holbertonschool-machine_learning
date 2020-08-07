#!/usr/bin/env python3
""" Agglomerative clustering using scikit """

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Agglomerative clustering
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: maximum cophenetic distance for all clusters
    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster
             indices for each data point
    """
    linkage = sch.linkage(X, method='ward')
    clss = sch.fcluster(linkage, t=dist, criterion='distance')
    plt.figure()
    sch.dendrogram(linkage, color_threshold=dist)
    plt.show()
    return clss
