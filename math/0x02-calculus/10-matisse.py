#!/usr/bin/env python3
"""function that derivative a polynomial f(x)"""


def poly_derivative(poly):
    """function to calculate a derivative of a poly"""

    if type(poly) is not list:
        return None
    elif not poly:
        return None
    elif len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    else:
        power = [i for i in range(len(poly))]
        df = [power[x] * poly[x] for x in range(1, len(poly))]
        if len(list(df)) == 1:
            return [0]
        else:
            return df
