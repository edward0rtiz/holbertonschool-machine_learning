#!/usr/bin/env python3
"""Script to converts a numeric vector into a one-hot matrix"""


import numpy as np


def one_hot_encode(Y, classes):
    """

    Args:
        Y:  class labels type int
        classes: nmax classes

    Returns: np.array one_hot_encode matrix

    """
    if Y.size is 0:
        return None
    elif type(classes) is not int:
        return None
    elif not isinstance(Y, np.ndarray):
        return None
    else:
        ohe = np.zeros((classes, Y.shape[0]))
        ohe[Y, np.arange(Y.size)] = 1
        return ohe
