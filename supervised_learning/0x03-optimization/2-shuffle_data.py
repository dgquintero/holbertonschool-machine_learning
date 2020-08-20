#!/usr/bin/env python3
"""shuffle_data function"""

import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices
    Arguments:
        X: input to shuffle shape(m, nx)
            m: number of data points
            nx: number of features
        Y: input to shuffle shape(m, nx)
            m: number of data points
            nx: number of features
    Returns: the shuffled X and Y
    """
    randomize = np.random.permutation(X.shape[0])
    input1 = X[randomize]
    input2 = Y[randomize]
    return input1, input2
