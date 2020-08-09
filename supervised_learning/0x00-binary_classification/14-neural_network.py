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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
            Arguments:
                X: input data shape(nx, m)
                Y: "true" labels vector of shape (1, number of examples)
                A1: the output of the hidden layer
                A2: the predicted output
                alpha: the learning rate
            Returns: updates __W1 __b1 __W2 __b2
        """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(A1, dZ2.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update rule for each parameter
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dW2).T
        self.__b2 = self.__b2 - (alpha * db2)

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
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        return self.evaluate(X, Y)
