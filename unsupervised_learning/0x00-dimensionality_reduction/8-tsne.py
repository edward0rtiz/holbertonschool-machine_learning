#!/usr/bin/env python3
"""t-sne"""

import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    T-sne function
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset to be
           transformed by t-SNE
        ndims: the new dimensional representation of X
        idims: the intermediate dimensional representation of X after
                PCA
        perplexity:  perplexity
        iterations: number of iterations
        lr: learning rate
    Returns: Y, a numpy.ndarray of shape (n, ndim) containing the optimized
             low dimensional
    """
    n, d = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8

    # min_gain = 0.01
    # gains = np.ones((n, ndims))

    X = pca(X, idims)
    P = P_affinities(X, perplexity=perplexity)
    Y = np.random.rand(n, ndims)
    #iY = np.zeros((n, ndims))
    iY = Y
    # early exaggeration
    P = P * 4.

    for i in range(iterations):

        dY, Q = grads(Y, P)
        if i < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        # delta-bar-delta algorithm for SDG optional
        """gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain"""

        #iY = momentum * iY - lr * (gains * dY)
        iY = momentum * iY - lr * dY
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            print('Cost at iteration {}: {}'.format((i+1), C))
        if i == 100:
            P = P / 4.

    return Y
