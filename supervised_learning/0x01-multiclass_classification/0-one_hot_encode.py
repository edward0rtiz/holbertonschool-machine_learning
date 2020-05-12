#!/usr/bin/python3
"""Script to converts a numeric vector into a one-hot matrix"""


import numpy as np

def one_hot_encode(Y, classes):

    if Y.size is 0:
        return None
    else:
        ohe = np.zeros((classes, Y.shape[0]), dtype=int)
        ohe[Y, np.arange(Y.size)] = 1
        return ohe