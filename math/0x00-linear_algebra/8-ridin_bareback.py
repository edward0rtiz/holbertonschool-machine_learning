#!/usr/bin/env python3
"""function to multiply 2 matrices"""


def mat_mul(mat1, mat2):
    """ multiply two matrices

    Args:
        mat1, mat2: Given matrices

    Return:
        the new mat: new_mat
    """
    if len(mat1[0]) != len(mat2):
        return None
    else:
        new_mat = []
        for i in range(len(mat1)):
            mat_i = []
            for j in range(len(mat2[0])):
                vec = 0
                for k in range(len(mat2)):
                    vec += mat1[i][k] * mat2[k][j]
                mat_i.append(vec)
            new_mat.append(mat_i)
        for x in new_mat:
            return new_mat
