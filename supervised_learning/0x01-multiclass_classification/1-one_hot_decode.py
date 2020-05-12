#!/usr/bin/python3
"""Script to converts a numeric vector into a one-hot matrix"""


import numpy as np

def one_hot_decode(one_hot):

    if not isinstance(one_hot, np.ndarray):
        return None
    else:
        return np.argmax(one_hot, axis=0)
