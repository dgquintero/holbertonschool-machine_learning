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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if layers[i] < 1 or type(layers[i]) is not int:
                raise TypeError("layers must be a list of positive integers")

            w_i = "W" + str(i + 1)
            b_i = "b" + str(i + 1)

            if i == 0:
                self.weights[w_i] = np.random.randn(layers[i], nx)\
                                            * np.sqrt(2 / nx)
            if i > 0:
                self.weights[w_i] = np.random.randn(layers[i], layers[i - 1])\
                    * np.sqrt(2 / layers[i - 1])
            self.weights[b_i] = np.zeros((layers[i], 1))