#!/usr/bin/env python3
"""Script to implement momentum algorithm"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Function to calculate momentum optimization
    Args:
        alpha: hyper-parameter learning rate
        beta1: momentum weight
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: previous first moment of var

    Returns:  the updated variable and the new moment, respectively

    """
    V = (beta1 * v) + ((1 - beta1) * grad)
    W = var - (alpha * V)
    return W, V
