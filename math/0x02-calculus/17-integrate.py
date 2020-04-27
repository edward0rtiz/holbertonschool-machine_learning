#!/usr/bin/env python3
"""function that integrates a poly f(x)"""


def poly_integral(poly, C=0):
    """function to integrate a poly"""

    if type(poly) is not list:
        return None
    elif not poly:
        return None
    elif len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    elif type(C)is not int:
        return None
    elif poly == [0]:
        return [C]
    else:
        integral = []
        integral.append(C)
        for i in range(len(poly)):
            x = poly[i]/(i + 1)
            integral.append(x if x.is_integer() else x)
        return integral
