#!/usr/bin/env python3
""" function to add two arrays elements-wise"""


def add_arrays(arr1, arr2):
    """ adding two arrays element wise

    Args:
        arr1, arr2: Given arrays

    Return:
        the sum of arrays: new matrix

    """
    if len(arr1) != len(arr2):
        return None
    else:
        return [sum(x) for x in zip(arr1, arr2)]
