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
