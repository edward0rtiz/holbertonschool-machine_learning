#!/usr/bin/env python3
"""Script to optimize DNN using RMSprop"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Function to optimize DNN
    Args:
        alpha: learning rate
        beta2: RMSProp weight
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        s: previous second moment of var

    Returns: updated variable and the new moment, respectively

    """
    Sdv = (beta2 * s) + ((1 - beta2) * grad ** 2)
    new_V = var - alpha * (grad / (Sdv ** (1 / 2) + epsilon))
    return new_V, Sdv
