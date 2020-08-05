#!/usr/bin/env python3

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
