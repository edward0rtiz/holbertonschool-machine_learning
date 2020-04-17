#!/usr/bin/env python3
"""function to add n dimension matrices with the same shape"""


def shape(matrix):
    if type(matrix[0]) != list:
        return [len(matrix)]
    else:
        return [len(matrix)] + shape(matrix[0])


def rec_matrix(mat1, mat2, depth, axis=0):
    new_mat = []

    if (type(mat1[0]) and type(mat2[0])) and depth == axis:
        new_mat = [y for x in [mat1, mat2] for y in x]
        return new_mat
    else:
        for x in range(len(mat1)):
            if type(mat1[x]) == list:
                new_mat = [sum(x) for x in zip(rec_matrix(mat1[x],
                                                          mat2[x],
                                                          depth + 1,
                                                          axis))]
            return new_mat


def cat_matrices(mat1, mat2, axis=0):
    if shape(mat1) != shape(mat2):
        return None
    else:
        depth = 0
        new_mat = rec_matrix(mat1, mat2, depth, axis)
        return new_mat
