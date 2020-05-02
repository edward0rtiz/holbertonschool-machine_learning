#!/usr/bin/env python3
"""
Script to calculate a Binomial distribution
"""


class Binomial():
    """
    Tye class to call methods of Binomial distribution
    CDF and PMF
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize method
        data: type list data elements
        n: int element to be evaluated
        p: Boolean value
        """

        self.n = int(n)
        self.p = float(p)

        if data is None:
            if self.n < 1:
                raise ValueError("n must be a positive value")
            elif self.p <= 0 or self.p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = (sum([(data[i] - mean) ** 2
                             for i in range(len(data))]) / len(data))
            self.p = 1 - (variance / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def pmf(self, k):
        """
        Method Probability Mass Function for binomial
        k: integer value of the data
        return: PMF
        """

        k = int(k)
        factor_k = 1
        factor_n = 1
        factor_c = 1
        if k > self.n or k < 0:
            return 0
        else:
            for i in range(1, k + 1):
                factor_k *= i
            for j in range(1, self.n + 1):
                factor_n *= j
            for f in range(1, (self.n - k) + 1):
                factor_c *= f
            comb = factor_n / (factor_c * factor_k)
            prob = (self.p ** k) * ((1 - self.p) ** (self.n - k))
            pmf = comb * prob
        return pmf

    def cdf(self, k):
        """
        Method Cumulative distribution function
        k: integer value of the data
        return: CFD
        """

        k = int(k)
        if k < 0:
            return 0
        else:
            cdf = 0
            for i in range(k + 1):
                cdf += self.pmf(i)
            return cdf
