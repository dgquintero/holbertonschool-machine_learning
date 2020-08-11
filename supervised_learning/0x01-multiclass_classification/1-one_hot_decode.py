#!/usr/bin/env python3
"""One-hot decode function"""
import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a vector of labels
    Arguments:
        one_hot: onehot matrix shape(classes, m)
    Return: a numpy.ndarray with shape (m, )
    containing the numeric labels for each example
    """
    if len(one_hot) == 0:
        return None
    elif not isinstance(one_hot, np.ndarray):
        return None
    elif len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)
