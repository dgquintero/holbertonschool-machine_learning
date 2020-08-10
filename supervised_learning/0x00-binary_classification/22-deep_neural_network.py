#!/usr/bin/env python3
"""defines a DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork():
    """ Class DeepNeuralNetwork"""

    def __init__(self, nx, layers):
        """
        Arguments:
            nx: number of inputs
            layers: the number of nodes in each layer
        Return: (weight, bias, Aactivate output)
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(self.L):
            if layers[i] < 1 or type(layers[i]) is not int:
                raise TypeError("layers must be a list of positive integers")

            w_i = "W" + str(i + 1)
            b_i = "b" + str(i + 1)

            if i == 0:
                self.__weights[w_i] = np.random.randn(layers[i], nx)\
                                            * np.sqrt(2 / nx)
            if i > 0:
                self.__weights[w_i] = np.random.randn(layers[i], layers[i-1])\
                    * np.sqrt(2 / layers[i - 1])
            self.__weights[b_i] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """
        Private instance getter attr
        """
        return self.__L

    @property
    def cache(self):
        """
        Private instance getter attr
        """
        return self.__cache

    @property
    def weights(self):
        """
         Private instance getter attr
        """
        return self.__weights

    def forward_prop(self, X):
        """public method that calculates the forward propagation
            of the Neural Network
                Arguments:
                    X: input data shape(nx, m)
                Returns: private attributes __A1 and __A2
        """
        self.__cache["A0"] = X
        for i in range(self.__L):
            weights = self.__weights
            cache = self.__cache
            w_i = "W" + str(i + 1)
            b_i = "b" + str(i + 1)
            Za = np.matmul(weights[w_i], cache["A" + str(i)])
            Z = Za + weights[b_i]
            cache["A" + str(i + 1)] = 1 / (1 + np.exp(-Z))
        return cache["A" + str(self.__L)], cache

    def cost(self, Y, A):
        """
        Public method that calculates the cost of the
        model using logistic regression
            Arguments:
                Y: "true" labels vector of shape (1, number of examples)
                A: The sigmoid output of the activation of
                each example of shape (1, number of examples)
            Returns: The cost
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            np.multiply(
                Y, np.log(A)) + np.multiply(
                1 - Y, np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """
        Public method taht evaluates the neural networks predictions
            Arguments:
                X: input data shape(nx, m)
                Y: "true" labels vector of shape (1, number of examples)
            Returns: the neuron’s prediction and the cost of the network
            respectively
        """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
            Arguments:
                X: input data shape(nx, m)
                Y: "true" labels vector of shape (1, number of examples)
                cache: Activated neurons in layers
                alpha: the learning rate
            Returns: updates __weights
        """
        m = Y.shape[1]
        dZ = cache['A'+str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dW = (1 / m) * np.matmul(cache['A' + str(i - 1)], dZ.T)
            dZ = np.matmul(self.__weights['W'+str(i)].T, dZ) *\
                (cache['A'+str(i-1)] * (1 - cache['A'+str(i-1)]))
            # update weiths and bias
            self.__weights['W'+str(i)] = self.__weights['W'+str(i)] -\
                (alpha * dW).T
            self.__weights['b'+str(i)] = self.__weights['b'+str(i)] -\
                (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains neural network updates __W1, __b1, __A1, __W2, __b2, __A2
            Arguments:
                X: input data shape(nx, m)
                Y: "true" labels vector of shape (1, number of examples)
                iterations: # of iterations
                alpha: learning rate
                verbose: boolean print or not information about training
                graph: is a boolean that defines whether or not to graph
                step: Boolean
            Return: evaluation of the training data after iterations
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            cache = self.__cache
            self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
        return self.evaluate(X, Y)
