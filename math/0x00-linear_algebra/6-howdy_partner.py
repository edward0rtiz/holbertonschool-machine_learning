#!/usr/bin/env python3
"""function to concatenate 2 arrays into a new one"""


def cat_arrays(arr1, arr2):
    """ concatenate two arrays into a new one

    Args:
        arr1, arr2: Given array

    Return:
        the new list of arrays: arr3

    """
    arr3 = []
    arr3 = arr1 + arr2
    return arr3
