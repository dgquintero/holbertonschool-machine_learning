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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for l in range(self.L):
            if layers[l] < 1 or type(layers[l]) is not int:
                raise TypeError("layers must be a list of positive integers")

            self.weights['b' + str(l + 1)] = np.zeros((layers[l], 1))
            if l == 0:
                self.weights['W' + str(l + 1)] = np.random.randn(layers[l], nx) * np.sqrt(2 / nx)            
            else:
                self.weights['W' + str(l + 1)] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / layers[l - 1])

