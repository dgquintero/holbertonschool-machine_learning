#!/usr/bin/env python3
"""defines a NeuralNetwork"""
import numpy as np


class NeuralNetwork():
    """ Class NeuralNetwork"""

    def __init__(self, nx, nodes):
        """
        Arguments:
            nx: number of inputs
            nodes: nodes found in the hidden layer
        Return: (weight, bias, Aactivate output)
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nx, nodes).reshape(nodes, nx)
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.randn(nodes).reshape(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ getter function for Weight """
        return self.__W1

    @property
    def b1(self):
        """ getter function for bias """
        return self.__b1

    @property
    def A1(self):
        """getter function for Activate output"""
        return self.__A1

    @property
    def W2(self):
        """ getter function for Weight """
        return self.__W2

    @property
    def b2(self):
        """ getter function for bias """
        return self.__b2

    @property
    def A2(self):
        """getter function for Activate output"""
        return self.__A2

    def forward_prop(self, X):
        """public method that calculates the forward propagation
            of the Neural Network
                Arguments:
                    X: input data shape(nx, m)
                Returns: private attributes __A1 and __A2"""
        Z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

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
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost
