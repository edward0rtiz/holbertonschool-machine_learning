# !/usr/bin/env python3
"""Class NeuralNetwork"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers, activation='sig'):
        """
        class constructor
        :param nx: is the number of input features
        :param layers: is a list representing the number of nodes in each
        layer of the network
        :param activation: represents the type of activation function used
        in the hidden layers
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        layers_num = np.array(layers)
        if np.any(layers_num < 1) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")
        for i in range(self.__L):
            key_w = "W{}".format(i + 1)
            key_b = "b{}".format(i + 1)
            if i == 0:
                weight = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights[key_w] = weight
            else:
                weight = np.random.randn(layers[i], layers[i - 1]) * \
                         np.sqrt(2 / layers[i - 1])
                self.__weights[key_w] = weight
            bias = np.zeros((layers[i], 1))
            self.__weights[key_b] = bias

    @property
    def activation(self):
        """
        Activation attribute getter
        :return:
        """
        return self.__activation

    @property
    def L(self):
        """
        L attribute getter
        :return: The number of layers in the neural network.
        """
        return self.__L

    @property
    def cache(self):
        """ cache attribute getter.
        :return: A dictionary to hold all intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """ weights attribute getter.
        :return: A dictionary to hold all weights and biased of the network.
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        :param X: numpy.ndarray with shape (nx, m) that contains the input data
                  nx is the number of input features to the neuron
                  m is the number of examples
        :return: the output of the neural network and the cache, respectively
        """
        self.__cache['A0'] = X
        for i in range(self.__L):
            key_w = "W{}".format(i + 1)
            key_b = "b{}".format(i + 1)
            key_a = "A{}".format(i)
            new_key_a = "A{}".format(i + 1)
            z = np.matmul(self.__weights[key_w], self.__cache[key_a]) \
                + self.__weights[key_b]
            if i == self.__L - 1:
                # Softmax
                t = np.exp(z)
                activation = np.exp(z) / t.sum(axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    # Sigmoid
                    activation = 1 / (1 + np.exp(-z))
                else:
                    # Tanh
                    activation = (np.exp(z) - np.exp(-z)) / \
                                 (np.exp(z) + np.exp(-z))
            self.__cache[new_key_a] = activation

        return activation, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        :param Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :param A: is a numpy.ndarray with shape (1, m) containing
        the activated output of the neuron for each example
        :return: the cost
        """
        summatory = Y * np.log(A)
        constant = -(1 / Y.shape[1])
        return constant * summatory.sum()

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        :param X: numpy.ndarray with shape (nx, m) that contains
        the input data
        :param Y: is a numpy.ndarray with shape (1, m) that
        contains
        the correct labels for the input data
        :return: the neuron’s prediction and the cost of the
        network
        """
        output, _ = self.forward_prop(X)
        cost = self.cost(Y, output)
        prediction = np.where(output == np.amax(output, axis=0), 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        :param Y: is a ndarray with shape (1, m) that contains
        the correct labels for the input data
        :param cache: is a dictionary containing all the intermediary
        values of the network
        :param alpha: is the learning rate
        :return: Updates the private attribute __weights
        """
        for i in reversed(range(self.__L)):
            key_w = "W{}".format(i + 1)
            key_b = "b{}".format(i + 1)
            key_a = "A{}".format(i + 1)
            A = cache[key_a]
            m = Y.shape[1]
            if i == self.__L - 1:
                dz = A - Y
                w = self.__weights[key_w]
            else:
                if self.__activation == 'sig':
                    g = A * (1 - A)
                elif self.__activation == 'tanh':
                    g = 1 - (A * A)
                part1 = np.matmul(w.T, dz)
                dz = part1 * g
                w = self.__weights[key_w]
            dw = np.matmul(cache["A{}".format(i)], dz.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights[key_w] = self.__weights[key_w] - (alpha * dw.T)
            self.__weights[key_b] = self.__weights[key_b] - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network
        :pa ram X: is a ndarray with shape (nx, m) that contains the input data
                  nx is the number of input features to the neuron
                  m is the number of examples
        :param Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :param iterations: is the number of iterations to train over
        :param alpha: is the learning rate
        :return: the evaluation of the training data after
        iterations of training have occurred
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        graph_iteration = []
        graph_cost = []

        for i in range(iterations + 1):
            output, cache = self.forward_prop(X)
            cost = self.cost(Y, output)

            if step and (i % step == 0 or i == iterations):
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                graph_iteration.append(i)
                graph_cost.append(cost)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph is True:
            plt.plot(graph_iteration, graph_cost)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        :param filename: is the file to which the object should be saved
        :return: Nothing
        """
        if '.pkl' not in filename:
            filename = filename + '.pkl'
        with open(filename, 'wb') as fd:
            pickle.dump(self, fd)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        :param filename: is the file from which the object should be loaded
        :return: the loaded object, or None if filename doesn’t exist
        """
        try:
            with open(filename, 'rb') as fd:
                file_object = pickle.load(fd)
                return file_object
        except FileNotFoundError:
            return None
