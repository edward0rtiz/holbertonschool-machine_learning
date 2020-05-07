#!/usr/bin/env python3
"""
Script to train a neuron including graphic
"""

import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    """ Class Neuron """

    def __init__(self, nx):
        """

        Args:
            nx: Type int the number of n inputs into the ANN
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(nx).reshape(1, nx)  # Weight
        self.__b = 0  # Bias
        self.__A = 0  # output

    @property
    def W(self):
        """

        Returns: private instance weight

        """
        return self.__W

    @property
    def b(self):
        """

        Returns: private instance bias

        """
        return self.__b

    @property
    def A(self):
        """

        Returns: private instance output

        """
        return self.__A

    def forward_prop(self, X):
        """
        Function of forward propagation
        activated by a sigmoid function

        Args:
            X: ndarray with shape of nx, m

        Returns: neuron activated using sigmoid

        """
        EWx = np.matmul(self.__W, X) + self.__b  # z (sum of weight and X's)
        z = EWx
        sigmoid = 1 / (1 + np.exp(-z))  # (σ): g(z) = 1 / (1 + e^{-z})
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """
        Cost function using binary cross-entropy
        Args:
            Y: Y hat, slope
            A: Activated neuron output

        Returns: Cost value, efficiency when C = 0

        """

        m = Y.shape[1]
        C = - (1 / m) * np.sum(
            np.multiply(
                Y, np.log(A)) + np.multiply(
                1 - Y, np.log(1.0000001 - A)))
        return C

    def evaluate(self, X, Y):
        """

        Args:
            X: input neuron, shape (nx, m)
            Y: Correct labels for the input data

        Returns: The neuron prediction and the cost
                of the network
        """

        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        prediction = np.where(self.__A >= 0.5, 1, 0)  # broadcasting
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """

        Args:
            X: input neuron, shape (nx, m)
            Y: Correct labels vector
            A: Activated neuron output
            alpha: learning rate

        Returns: gradient descent bias + adjusted weights

        """

        m = Y.shape[1]
        dz = A - Y  # derivative z
        dW = np.matmul(X, dz.T) / m  # grad of the loss with respect to w
        db = np.sum(dz) / m  # grad of the loss with respect to b
        self.__W -= (alpha * dW).T
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """

        Args:
            step: Boolean of iterations in the model
            graph: Boolean of value of iterations against cost
            verbose: Boolean of string text print of cost
            X: input neuron, shape (nx, m)
            Y: Correct labels vector
            iterations: number of iterations to optimize the parameters
            alpha: learning rate

        Returns: output optimized and cost of training

        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        steps = 0
        c_ax = np.zeros(iterations + 1)

        for i in range(iterations):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A)
            if i == iterations and step:
                print("Cost after {} iterations: {}".format(i, cost))
                steps += step
            if verbose is True:
                print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(X, Y, self.__A, alpha)
            if graph is True:
                c_ax[i] = cost
        if graph is True:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(np.arange(0, iterations + 1), c_ax)
            plt.show()
        return self.evaluate(X, Y)
