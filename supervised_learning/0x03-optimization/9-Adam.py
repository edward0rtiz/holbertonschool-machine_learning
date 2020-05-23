#!/usr/bin/env python3
"""Script to optimize DNN using Adam"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    function to optimize DNN using Adam alogrithm
    Args:
        alpha: learning rate
        beta1: weight used for the first moment
        beta2: weight used for the second moment
        epsilon: small number to avoid division by zero
        var: containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: is the previous first moment of var
        s: previous second moment of var
        t: time step used for bias correction

    Returns: updated variable, the new first moment,
            and the new second moment, respectively

    """
    V = (beta1 * v) + ((1 - beta1) * grad)
    V_corrected = V / (1 - beta1 ** t)
    S = (beta2 * s) + ((1 - beta2) * grad ** 2)
    S_corrected = S / (1 - beta2 ** t)

    up_var = var - alpha * (V_corrected / ((S_corrected ** (1 / 2)) + epsilon))
    return up_var, V, S
