#!/usr/bin/env python3
"""
Script to calculate a Exponential distribution
"""


class Exponential():
    """
    Tye class to call methods of Exponential distribution
    CDF and PDF
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ init """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = (1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Method Probability Density function
        k: integer value of the data
        return: PDF
        """
        if x < 0:
            return 0
        else:
            pdf = self.lambtha * (Exponential.e**((-self.lambtha) * x))
            return pdf

    def cdf(self, x):
        """
        Method Cumulative distribution function
        k: integer value of the data
        return: CDF
        """
        if x < 0:
            return 0
        return 1 - (Exponential.e**((-self.lambtha) * x))
