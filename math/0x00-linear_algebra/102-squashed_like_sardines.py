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


def rec_matrix(mat1, mat2, rank, axis=0):
    """ recursively operate a concatenation of a n matrix

        Args:
            mat1, mat2: Given matrix
            axis: Given axis
            rank: Given rank to check if it is in the same
            axis mat1 and mat2

        Return:
            the concatenation of mat1, mat2 iterating recursively: ndim
        """
    new_mat = []
    if (type(mat1[0]) and type(mat2[0])) and rank == axis:
        new_mat = [y for x in [mat1, mat2] for y in x]
        return new_mat
    else:
        for x in range(len(mat1)):
            if type(mat1[x]) == list:
                new_mat = [sum(x) for x in zip(rec_matrix(mat1[x],
                                                          mat2[x],
                                                          rank + 1,
                                                          axis))]
            return new_mat


def cat_matrices(mat1, mat2, axis=0):
    """ concatenate n dimesnional matrices with the same shape

    Args:
        mat1, mat2: Given matrix
        axis: given axis

    Return:
        new_mat: Recursively concatenation of mat1 and mat2

    """
    if shape(mat1) != shape(mat2):
        return None
    else:
        rank = 0
        new_mat = rec_matrix(mat1, mat2, rank, axis)
        return new_mat
