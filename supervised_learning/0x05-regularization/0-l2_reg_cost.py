#!/usr/bin/env python3
""" l2_reg_cost function"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates the cost of a neural network with L2 regularization
    Arguments:
        cost:the cost of the network without L2 regularization
        lambtha: the regularization parameter
        weights: a dictionary of the weights and biases of the neural network
        L: is the number of layers in the neural network
        m: is the number of data points used
    Returns: the cost of the network accounting for L2 regularization
    """
    euc_norm = 0
    for key, values in weights.items():
        if key[0] == 'W':
            euc_norm += np.linalg.norm(values)
    cost_L2 = cost + (lambtha / (2 * m) * euc_norm)
    return cost_L2
