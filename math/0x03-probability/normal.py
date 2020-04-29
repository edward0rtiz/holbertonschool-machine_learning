#!/usr/bin/env python3
"""
Script to calculate a Normal distribution
of continous variables and discretes
"""


class Normal():
    """
    Type class normal distribution
    """

    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        data: list of data given
        mean: self attribute of the mean of the data
        stddev: standard error of the data
        """

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            sigma = 0
            for i in range(0, len(data)):
                x = (data[i] - self.mean) ** 2
                sigma += x
            self.stddev = (sigma / len(data)) ** (1 / 2)

    def z_score(self, x):
        """
        x: x_value of the function
        return: the z_score
        """

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        z: z value of the x value replica
        return: x_value
        """
        return self.stddev * z + self.mean

    def pdf(self, x):
        """
        x: the x parameter of the function
        return: Probability Density Function
        """

        p1 = 1 / (self.stddev * ((2 * Normal.pi) ** 0.5))
        p2 = ((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        return p1 * Normal.e ** (-p2)

    def cdf(self, x):
        """
        param x: x parameter of the function
        return:  Cumulative distribution Function
        """
        xa = (x - self.mean) / ((2 ** 0.5) * self.stddev)
        errof = (((4 / Normal.pi) ** 0.5) * (xa - (xa ** 3) / 3 +
                                             (xa ** 5) / 10 - (xa ** 7) / 42 +
                                             (xa ** 9) / 216))
        cdf = (1 + errof) / 2
        return cdf
