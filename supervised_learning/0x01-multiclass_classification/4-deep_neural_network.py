#!/usr/bin/env python3
"""Script to train model and plotting
    DNN
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    """
    Class Deep Neural Network
    """

    def __init__(self, nx, layers, activation='sig'):
        """

        Args:
            nx: input value
            nodes: nodes placed in the hidden layer
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation != "sig" and activation != "tanh":
            raise ValueError("activation must be 'sig or 'tanh")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        for lay in range(self.L):
            if layers[lay] <= 0 or type(layers[lay]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["b" + str(lay + 1)] = np.zeros((layers[lay], 1))
            if lay == 0:
                He_val = np.random.randn(layers[lay], nx) * np.sqrt(2 / nx)
                self.__weights["W" + str(lay + 1)] = He_val
            if lay > 0:
                He_val1 = np.random.randn(layers[lay], layers[lay - 1])
                He_val2 = np.sqrt(2 / layers[lay - 1])
                He_val3 = He_val1 * He_val2
                self.__weights["W" + str(lay + 1)] = He_val3

    @property
    def L(self):
        """
        Getter attr

        Returns: Private instance number of layers

        """
        return self.__L

    @property
    def cache(self):
        """
        Getter attr

        Returns: Private instance decit that hold intermediates
        values of the network

        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter attr
        Returns: Private instance holds weights and biases

        """
        return self.__weights

    @property
    def activation(self):
        """
        Getter attr
        Returns: Private instance holds activation function

        """
        return self.__activation

    def forward_prop(self, X):
        """
        Forward propagation function
        Args:
            X: x numpy array with shape (nx, m)

        Returns: forward propagation

        """
        self.__cache["A0"] = X
        for lay in range(self.__L):
            weights = self.__weights
            cache = self.__cache
            activ = self.__activation
            Za = np.matmul(weights["W" + str(lay + 1)], cache["A" + str(lay)])
            Z = Za + weights["b" + str(lay + 1)]
            if lay == self.__L - 1:
                t = np.exp(Z)
                # softmax activation
                cache["A" + str(lay + 1)] = (t / np.sum(
                    t, axis=0, keepdims=True))
            else:
                if activ == 'sig':
                    cache["A" + str(lay + 1)] = 1 / (1 + np.exp(-Z))
                else:
                    cache["A" + str(lay + 1)] = np.tanh(Z)
        return cache["A" + str(lay + 1)], cache

    def cost(self, Y, A):
        """
        Cost function using binary cross-entropy
        Args:
            Y: Y hat, slope
            A: Activated neuron output

        Returns: Cost value, efficiency when C = 0

        """

        m = Y.shape[1]
        C = (-1 / m) * np.sum(Y * np.log(A))
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
        cache = self.__cache
        cost = self.cost(Y, cache["A" + str(self.__L)])
        mc = np.amax(cache["A" + str(self.__L)], axis=0)
        # broadcasting
        prediction = np.where(cache["A" + str(self.__L)] == mc, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """

        Args:
            X: input neuron, shape (nx, m)
            Y: Correct labels vector
            cache: Activated neurons in n layer
            alpha: learning rate

        Returns: gradient descent bias + adjusted weights

        """
        m = Y.shape[1]
        tW = self.__weights.copy()
        for i in reversed(range(self.__L)):
            A = self.__cache["A" + str(i + 1)]
            if i == self.__L - 1:
                dZ = self.__cache["A" + str(i + 1)] - Y
                dW = np.matmul(self.__cache["A" + str(i)], dZ.T) / m
            else:
                dW2 = np.matmul(tW["W" + str(i + 2)].T, dZ2)
                if self.__activation == 'sig':
                    gd = A * (1 - A)
                elif self.__activation == 'tanh':
                    gd = 1 - (A * A)
                dZ = dW2 * gd
                dW = np.matmul(dZ, self.__cache["A" + str(i)].T) / m
            # grad of the loss with respect to b
            db3 = np.sum(dZ, axis=1, keepdims=True) / m
            if i == self.__L - 1:
                self.__weights["W" + str(i + 1)] = (tW["W" +
                                                       str(i + 1)] -
                                                    (alpha * dW).T)
            else:
                self.__weights["W" + str(i + 1)] = (tW["W" +
                                                       str(i + 1)] -
                                                    (alpha * dW))
            self.__weights["b" + str(i + 1)] = tW["b" + str(i + 1)] - (
                    alpha * db3)
            dZ2 = dZ

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
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

        temp_cost = []
        temp_iterations = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__cache["A" + str(self.__L)])
            if i % step == 0 or i == iterations:
                temp_cost.append(cost)
                temp_iterations.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph is True:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(temp_iterations, temp_cost)
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """

        Args:
            filename: pickle file

        Returns: saved object

        """
        if '.pkl' not in filename:
            filename += '.pkl'

        fileObject = open(filename, 'wb')
        pickle.dump(self, fileObject)
        fileObject.close()

    @staticmethod
    def load(filename):
        """

        Args:
            filename: pickle file

        Returns: Objects loaded

        """
        try:
            with open(filename, 'rb') as f:
                fileOpen = pickle.load(f)
            return fileOpen
        except FileNotFoundError:
            return None
