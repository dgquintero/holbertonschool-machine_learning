#!/usr/bin/env python3
"""update_variables_RMSProp function"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    that updates a variable using the RMSProp optimization algorithm
    Arguments:
        alpha: is the learning rate
        beta2: isnthe RMSProp weight
        epsilon is a small number to avoid division by zero
        var: ndarray containing the variable to be updated
        grad: ndarray containing the gradient of var
        s: is the previous second moment of var
    Returns: the momentum optimization operation
    """
    Sdw = (beta2 * s) + ((1 - beta2) * grad ** 2)
    W = var - alpha * (grad / (Sdw ** (1 / 2) + epsilon))
    return W, Sdw
