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
    m = Y.shape[1]
    W_copy = weights.copy()
    for i in range(L, 0, -1):
        if i == L - 1:
            dZ = cache['A' + str(i + 1)] - Y
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dW = (1 / m) * np.matmul(cache['A' + str(i - 1)], dZ.T).T
            dW_L2 = dW + (lambtha / m) * W_copy['W' + str(i + 1)]
        else:
            dw2 = np.matmul(W_copy['W' + str(i)].T, dZ2)
            tanh = 1 - (cache['A' + str(i + 1)] ** 2)
            dZ = dw2 * tanh
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dW = (1 / m) * np.matmul(dZ, cache['A' + str(i)].T)
            dW_L2 = dW + (lambtha / m) * W_copy['W' + str(i + 1)]
        # update weiths and bias
        weights['W' + str(i+1)] = (W_copy['W'+str(i+1)] -\
            (alpha * dW_L2)).T
        weights['b' + str(i+1)] = W_copy['b'+str(i + 1)] -\
            (alpha * db)
        dZ2 = dZ
