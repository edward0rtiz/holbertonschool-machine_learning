#!/usr/bin/env python3
"""
Determinant function
"""


def multi_determinant(matrix):
    """
    function that computes the determinant of a given matrix of
    +3D
    Args:
        matrix: list of lists whose determinant should be calculated
    Returns: Determinant of matrix
    """
    mat_l = len(matrix)
    if mat_l == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    deter = 0
    cols = list(range(len(matrix)))
    for c in cols:
        mat_cp = [r[:] for r in matrix]
        mat_cp = mat_cp[1:]
        rows = range(len(mat_cp))

        for r in rows:
            mat_cp[r] = mat_cp[r][0:c] + mat_cp[r][c + 1:]
        sign = (-1) ** (c % 2)
        sub_det = multi_determinant(mat_cp)
        deter += sign * matrix[0][c] * sub_det
    return deter


def determinant(matrix):
    """
    calculates the determinant of a matrix
    Args:
        matrix: list of lists whose determinant should be calculated
    Returns: Determinant of matrix
    """
    mat_l = len(matrix)
    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("Matrix must be a list of list")
    if matrix[0] and mat_l != len(matrix[0]):
        raise ValueError("Matrix must be a square matrix")
    if matrix == [[]]:
        return 1
    if mat_l == 1:
        return matrix[0][0]
    if mat_l == 2 and len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    return multi_determinant(matrix)


