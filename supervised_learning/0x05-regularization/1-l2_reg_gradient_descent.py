""" l2_reg_cost function"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    calculates the cost of a neural network with L2 regularization
    Arguments:
        Y: Correct labels for the data shape(classes, m)
        cache: dictionary of the outputs of each layer of the NN
        alpha: the learning rate
        lambtha: the regularization parameter
        weights: a dictionary of the weights and biases of the neural network
        L: is the number of layers in the neural network
        m: is the number of data points used
    Returns: the cost of the network accounting for L2 regularization
    """
    

