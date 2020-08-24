#!/usr/bin/env python3
"""
Bayes Optimization
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    Bayes Optimization using Gaussian Process
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01,
                 minimize=True):
        """
        * f is the black-box function to be optimized
        * X_init is a numpy.ndarray of shape (t, 1) representing
          the inputs already sampled with the black-box function
        * Y_init is a numpy.ndarray of shape (t, 1) representing
          the outputs of the black-box function for each input
          in X_init
          - t is the number of initial samples
        * bounds is a tuple of (min, max) representing the bounds
          of the space in which to look for the optimal point
        * ac_samples is the number of samples that should be analyzed
          during acquisition
        * l is the length parameter for the kernel
        * sigma_f is the standard deviation given to the output of
          the black-box function
        * xsi is the exploration-exploitation factor for acquisition
        * minimize is a bool determining whether optimization should
          be performed for minimization (True) or maximization (False)
        * Sets the following public instance attributes:
          - f: the black-box function
          - gp: an instance of the class GaussianProcess
          - X_s: a numpy.ndarray of shape (ac_samples, 1) containing
            all acquisition sample points, evenly spaced between min
            and max
          - xsi: the exploration-exploitation factor
          - minimize: a bool for minimization versus maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        X_s = np.linspace(min, max, ac_samples)
        self.X_s = (np.sort(X_s)).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        * Uses the Expected Improvement acquisition function
        Returns: X_next, EI
        * X_next is a numpy.ndarray of shape (1,) representing the next best
          sample point
        * EI is a numpy.ndarray of shape (ac_samples,) containing the expected
          improvement of each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            optimize = np.amin(self.gp.Y)
            imp = optimize - mu - self.xsi

        else:
            optimize = np.amax(self.gp.Y)
            imp = mu - optimize - self.xsi

        Z = np.zeros(sigma.shape[0])

        for i in range(sigma.shape[0]):
            if sigma[i] != 0:
                Z[i] = imp[i] / sigma[i]
            else:
                Z[i] = 0

        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

        index = np.argmax(ei)
        best_sample = self.X_s[index]

        return (best_sample, ei)

    def optimize(self, iterations=100):
        """
        Optimize method
        Args:
            iterations: maximum number of iterations to perform
        Returns: x_opt, y_opt
                 x_opt: numpy.ndarray of shape (1,) representing the optimal
                 point
                 y_opt: numpy.ndarray of shape (1,) representing the optimal
                 function value
        """

        X_all_s = []
        for i in range(iterations):
            # Find the next sampling point xt by optimizing the acquisition
            # function over the GP: xt = argmaxx μ(x | D1:t−1)

            x_opt, _ = self.acquisition()
            # If the next proposed point is one that has already been sampled,
            # optimization should be stopped early
            if x_opt in X_all_s:
                break

            y_opt = self.f(x_opt)

            # Add the sample to previous samples
            # D1: t = {D1: t−1, (xt, yt)} and update the GP
            self.gp.update(x_opt, y_opt)
            X_all_s.append(x_opt)

        if self.minimize is True:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)

        self.gp.X = self.gp.X[:-1]

        x_opt = self.gp.X[index]
        y_opt = self.gp.Y[index]

        return x_opt, y_opt
