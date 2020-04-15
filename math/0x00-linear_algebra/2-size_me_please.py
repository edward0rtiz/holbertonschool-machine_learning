#!/usr/bin/env python3
""" Recursion function to check the shape of the matrix"""


def matrix_shape(matrix):
    """ return the shape of a matrix """
    if type(matrix[0]) != list:
        return [len(matrix)]
    else:
        return matrix_shape(matrix[0]) + [len(matrix)]
