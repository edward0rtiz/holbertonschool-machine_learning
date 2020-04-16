#!/usr/bin/env python3
"""function to slice matrix along specific values"""


import numpy as np


def np_slice(matrix, axes={}):
    """ slice matrix along specific value

    Args:
        matrix: Given matrix

    Return:
        the slice mat: slice_mat

    """
    slice_mat = [slice(None)] * matrix.ndim
    for k, v in sorted(axes.items()):
        slice_val = slice(*v)
        slice_mat[k] = slice_val
    matrix = matrix[tuple(slice_mat)]
    return matrix
