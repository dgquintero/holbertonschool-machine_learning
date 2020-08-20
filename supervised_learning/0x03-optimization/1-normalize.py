#!/usr/bin/env python3
"""normalize function"""


def normalize(X, m, s):
    """
    normalizes (standardizes) a matrix
    Arguments:
        X: input to normalize shape(d, nx)
            d: number of data points
            nx: number of features
        m: the mean of all features of X shape(nx,)
        s: standard deviation of all features of X shape(nx,)
    Returns: matrix normalized
    """
    return (X - m) / s
