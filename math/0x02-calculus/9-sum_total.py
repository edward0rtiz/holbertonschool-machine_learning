#!/usr/bin/env python3
"""function that sum powers based on Faulhaber's Formula"""


def summation_i_squared(n):
    """m(m+1)(2m+1) / 6"""

    if type(n) is not int:
        return None
    else:
        return int((n * (1 + n)) * (2 * n + 1) / 6)
