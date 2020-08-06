#!/usr/bin/env python3
"""defines a single neuron performing binary classification"""
import numpy as np


class Neuron():
    """ defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """nx: number of inputs
        (weight, bias, Aactivate output) """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ getter function for Weight """
        return self.__W

    @property
    def b(self):
        """ getter function for bias """
        return self.__b

    @property
    def A(self):
        """getter function for Activate output"""
        return self.__A

    def forward_prop(self, X):
        """ public method that calculates the forward propagation
            of the neuron
        """
        Z = np.matmul(self.W, X) + self.b
        sig = 1 / (1 + np.exp(-Z))
        self.__A = sig
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using
        logistic regression"""
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            np.multiply(
                Y, np.log(A)) + np.multiply(
                1 - Y, np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron
            Arguments:
                X: input neuron, shape (nx, m)
                Y: Correct labels vector
                A: Activated neuron output
                alpha: learning rate
            Return: Updates the private attributes __W and __b
        """
        m = Y.shape[1]
        dz = A - Y
        dW = (1 / m) * np.matmul(X, dz.T)
        db = (1 / m) * np.sum(dz)
        self.__W -= 1 * (alpha * dW).T
        self.__b -= 1 * (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron updates __w, __b __A
            Arguments:
                X: input neuron, shape (nx, m)
                Y: Correct labels vector, shape (1, m)
                iterations: # of iterations
                alpha: learning rate
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
            self.gradient_descent(X, Y, self.__A, alpha)
        return self.evaluate(X, Y)
