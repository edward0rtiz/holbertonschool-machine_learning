#!/usr/bin/env python3
""" Bayesian Information Criterion """

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    BIC function
    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing the minimum number of clusters
              to check for (inclusive)
        kmax: positive integer containing the maximum number of clusters
              to check for (inclusive)
        iterations: Positive integer containing the maximum number of
                    iterations for the EM algorithm
        tol: non-negative float containing the tolerance for the EM algorithm
        verbose: boolean that determines if the EM algorithm should print
                 information to the standard output
    Returns: best_k, best_result, l, b, or None, None, None, None on failure
             best_k: best value for k based on its BIC
             best_result : tuple containing pi, m, S
                           pi: numpy.ndarray of shape (k,) containing the
                               luster priors for the best number of clusters
                           m: numpy.ndarray of shape (k, d) containing the
                              centroid means for the best number of clusters
                           S: numpy.ndarray of shape (k, d, d) containing the
                              covariance matrices for the best number of
                              clusters
             l: numpy.ndarray of shape (kmax - kmin + 1) containing the log
                likelihood for each cluster size tested
             b: numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
                value for each cluster size tested
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None
    if type(kmax) != int or kmax <= 0 or kmax >= X.shape[0]:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None
    if type(tol) != float or tol <= 0:
        return None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None

    k_best = []
    best_res = []
    logl_val = []
    bic_val = []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S, log_l, _ = expectation_maximization(X, k, iterations, tol,
                                                      verbose)
        k_best.append(k)
        best_res.append((pi, m, S))
        logl_val.append(log_l)

        # Formula pf paramaters: https://bit.ly/33Cw8lH
        # code based on gaussian mixture source code n_parameters source code
        cov_params = k * d * (d + 1) / 2.
        mean_params = k * d
        p = int(cov_params + mean_params + k - 1)

        # Formula for this task BIC = p * ln(n) - 2 * l
        BIC = p * np.log(n) - 2 * log_l
        bic_val.append(BIC)
    bic_val = np.array(bic_val)
    best_val = np.argmin(bic_val)
    logl_val = np.array(logl_val)

    k_best = k_best[best_val]
    best_res = best_res[best_val]

    return k_best, best_res, logl_val, bic_val
