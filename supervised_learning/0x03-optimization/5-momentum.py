#!/usr/bin/env python3
"""update_variables_momentum function"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    calculate momentum optimization
    Arguments:
        alpha: is the learning rate
        beta1: is the momentum weight
        var: containing the variable to be updated
        grad: ndarray containing the gradient of var
        v: is the previous first moment of var
    Returns: the updated variable and the new moment, respectively
    """
    V = (beta * v) + ((1 - beta1) * grad)
    W = var - (alpha * V)
    return W, V
