#!/usr/bin/env python3
"""
matrix of cofactors
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
    return multi_determinant(matrix)


def minor_val(matrix, idx_r, idx_c):
    """
    function that computes minor in each idx position of the given matrix
    Args:
        matrix: given matrix
        idx_r: row skipped
        idx_c: col skipped
    Returns: determinant of the matrix with row and col skipped
    """
    minor_mat = [rows[:idx_c] + rows[idx_c + 1:]
                 for rows in (matrix[:idx_r] + matrix[idx_r + 1:])]
    return determinant(minor_mat)


def cofactor(matrix):
    """
    Compute the matrix of cofactors
    Args:
        matrix: list of lists whose determinant should be calculated
    Returns: Minor of a matrix
    """
    mat_l = len(matrix)
    range_mat_l = range(len(matrix))

    if type(matrix) != list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([type(mat) == list for mat in matrix]):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")
    if matrix[0] and len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(mat_l == len(col) for col in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    if mat_l == 1 and len(matrix[0]) == 1:
        return [[1]]
    if mat_l == 2 and len(matrix[0]) == 2:
        # apply a "checkerboard" of minuses to the "Matrix of Minors" +-+-
        return [[matrix[1][1], -matrix[1][0]], [-matrix[0][1], matrix[0][0]]]

    minor_values = []
    for row in range_mat_l:
        minor_r = []
        for col in range_mat_l:
            minor_c = minor_val(matrix, row, col)
            # apply a "checkerboard" of minuses to the "Matrix of Minors" +-+-
            sign = (-1) ** (row + col)
            minor_r.append(minor_c * sign)
        minor_values.append(minor_r)
    return minor_values
