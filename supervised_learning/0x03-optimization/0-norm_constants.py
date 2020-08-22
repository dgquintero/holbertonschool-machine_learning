#!/usr/bin/env python3
"""normalization_constants function"""

import numpy as np


def normalization_constants(X):
    """
    calculates the normalization constants of a matrix
    Arguments:
        X: input to normalize shape(m, nx)
            m: number of data points
            nx: number of features
    Returns: mean and standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    return mean, stdev
