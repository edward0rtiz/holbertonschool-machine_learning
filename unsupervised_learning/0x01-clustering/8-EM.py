#!/usr/bin/env python3
"""
Final step iterate E and M to get EM algorithm
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X,
                             k,
                             iterations=1000,
                             tol=1e-5,
                             verbose=False):
    """
    EM in a GMM
    Arg:
        X: np.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of
                    iterations for the algorithm
        tol: non-negative float containing tolerance of the log likelihood,
             used to determine early stopping i.e. if the difference is
             less than or equal to tol you should stop the algorithm
        verbose: boolean that determines if you should print information
                 about the algorithm
    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
        pi: np.ndarray of shape (k,) containing the priors for each
            cluster
        m: np.ndarray of shape (k, d) containing the centroid means for
           each cluster
        S: np.ndarray of shape (k, d, d) containing the covariance matrices
           for each cluster
        g: np.ndarray of shape (k, n) containing the probabilities for
           each data point in each cluster
        l: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    loglikelihood = 0

    for i in range(iterations):
        g, loglikelihood_new = expectation(X, pi, m, S)
        if verbose is True and (i % 10 == 0):
            print("Log Likelihood after {} iterations: {}".format(
                i, loglikelihood_new.round(5)))
        if abs(loglikelihood_new - loglikelihood) < tol:
            break
        pi, m, S = maximization(X, g)
        loglikelihood = loglikelihood_new
    g, loglikelihood_new = expectation(X, pi, m, S)
    if verbose is True:
        print("Log Likelihood after {} iterations: {}".format(
            i + 1, loglikelihood_new.round(5)))
    return pi, m, S, g, loglikelihood_new
