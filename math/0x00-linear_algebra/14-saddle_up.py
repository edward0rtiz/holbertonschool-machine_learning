#!/usr/bin/env python3
"""function to multiply 2 matrices"""


import numpy as np


def np_matmul(mat1, mat2):
    """ multiply two matrices

    Args:
        mat1, mat2: Given matrices

    Return:
        the new mat: new_mat

    """
    new_mat = np.dot(mat1, mat2)
    return new_mat
