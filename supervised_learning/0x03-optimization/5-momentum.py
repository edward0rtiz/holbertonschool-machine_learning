#!/usr/bin/env python3
"""Script to implement momentum algorithm"""


def update_variables_momentum(alpha, beta1, var, grad, v):

    V = (beta1 * v) + ((1 + beta1) * grad)
    W = var - (alpha * V)
    return W, V
