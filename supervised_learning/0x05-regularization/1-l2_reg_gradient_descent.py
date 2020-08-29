""" l2_reg_gradient_descent function"""

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
    m = Y.shape[1]
    dz = cache['A'+str(L)] - Y
    for i in range(L, 0, -1):
        cost_L2 = (lambtha / m) * weights['W'+str(i)]
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dW = ((1 / m) * np.matmul(dz, cache['A'+str(i-1)].T)) + cost_L2
        dz = np.matmul(weights['W'+str(i)].T, dz) *\
            ((1 - cache['A'+str(i-1)] ** 2))
        weights['W'+str(i)] = weights['W'+str(i)] -\
            (alpha * dW)
        weights['b'+str(i)] = weights['b'+str(i)] -\
            (alpha * db)
