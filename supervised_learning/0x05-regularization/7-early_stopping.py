#!/usr/bin/env python3
"""early_stopping function"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Function that determines if you should stop gradient descent early
    Arguments:
        cost: the current validation cost of the neural network
        opt_cost: the lowest recorded validation cost of the neural network
        threshold: threshold used for early stopping
        patience: the patience count used for early stopping
        count: count of how long the threshold has not been met
    Returns: whether the network should be stopped early
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count != patience:
        return False, count
    else:
        return True, count
