#!/usr/bin/env python3
"""update_varupdate_variables_Adamiables_RMSProp function"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    that updates a variable using the Adam optimization algorithm
    Arguments:
        alpha: is the learning rate
        beta1: is the weight used for the first moment
        beta2: is the weight used for the second moment
        epsilon is a small number to avoid division by zero
        var: ndarray containing the variable to be updated
        grad: ndarray containing the gradient of var
        s: is the previous second moment of var
        t: is the time step used for bias correction
    Returns: the updated variable, the new first moment,
             and the new second moment, respectively
    """
    V = (beta1 * v) + ((1 - beta1) * grad)
    new_v = V / (1 - beta1 ** t)
    S = (beta2 * s) + ((1 - beta2) * grad ** 2)
    new_s = S / (1 - beta2 ** t)
    W = var - alpha * (new_v / ((new_s ** (1 / 2)) + epsilon))
    return W, V, S
