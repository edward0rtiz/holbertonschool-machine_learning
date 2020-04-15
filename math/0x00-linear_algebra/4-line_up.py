#!/usr/bin/env python3
""" Transpose function to check a new matrix transposed"""


def add_arrays(arr1, arr2):
    """ adding two arrays element wise

    Args:
        arr1, arr2: Given arrays

    Return:
        the sum of arrays: Transposed matrix

    """
    if len(arr1) != len(arr2):
        return None
    else:
        return [sum(x) for x in zip(arr1, arr2)]
