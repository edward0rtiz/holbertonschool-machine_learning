#!/usr/bin/env python3
"""function for basic operations of matrices using numpy"""


def np_elementwise(mat1, mat2):
    """ operate matrices

    Args:
        mat1, mat2: Given matrix

    Return:
        matrix operation
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
