#!/usr/bin/env python3
""" l2_reg_cost function"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Function that updates the weights of a neural network with
    Dropout regularization using gradient descent
    Arguments:
        Y: ndarray containing the correct labels shape(classes, m)
        weights: dictionary of the weights and biases of the NN
        L: the number of layers in the network
        cache: is a dictionary of the outputs and dropout masks of
            each layer of the neural network
        keep_prob: is the probability that a node will be kept
    Returns:
    """
    m = Y.shape[1]
    dAl = cache['A' + str(L)] - Y
    for i in reversed(range(1, L + 1)):
        W = 'W' + str(i)
        b = 'b' + str(i)
        A = 'A' + str(i - 1)
        D = 'D' + str(i - 1)
        dw = (np.matmul(cache[A], dAl.T) / m)
        db = (np.sum(dAl, axis=1, keepdims=True) / m)
        if i - 1 > 0:
            dAl = np.matmul(weights['W' + str(
                i)].T, dAl) * (1 - (cache[A]**2)) * (cache[D]/keep_prob)
        weights[W] = weights['W' + str(
            i)] - (alpha * dw).T
        weights[b] = weights['b' + str(
            i)] - (alpha * db)
