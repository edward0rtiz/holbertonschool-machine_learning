#!/usr/bin/env python3
"""function to add n dimension matrices with the same shape"""


def shape(matrix):
    """ return the shape of a matrix

    Args:
        matrix: Given matrix

    Return:
        the shape of the matrix: ndim

    """
    if type(matrix[0]) != list:
        return [len(matrix)]
    else:
        return [len(matrix)] + shape(matrix[0])


def rec_matrix(mat1, mat2):
    """ recursively operate an add of a n matrix

        Args:
            mat1, mat2: Given matrix

        Return:
            the addition of mat1, mat2 iterating recursively: ndim
        """
    new_mat = []

    if (type(mat1) and type(mat2)) == list:
        for i in range(len(mat1)):
            if type(mat1[i]) == list:
                new_mat.append(rec_matrix(mat1[i], mat2[i]))
            else:
                new_mat.append(mat1[i] + mat2[i])
        return new_mat


def add_matrices(mat1, mat2):
    """ add n dimesnional matrices with the same shape

    Args:
        mat1, mat2: Given matrix

    Return:
        new_mat: Recursively addition of mat1 and mat2

    """
    if shape(mat1) != shape(mat2):
        return None
    else:
        new_mat = rec_matrix(mat1, mat2)
        return new_mat
