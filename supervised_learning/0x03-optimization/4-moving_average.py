#!/usr/bin/env python3
"""shuffle_data function"""

import numpy as np


def moving_average(data, beta):
    """
    that calculates the weighted moving average of a data set
    Arguments:
        data: the list of data to calculate the moving average
        beta:  weight used for the moving average
    Returns: list containing the moving averages of data
    """
    vt = 0
    ma = []
    for i in range(len(data)):
        vt = (beta * vt) + ((1 - beta) * data[i])
        new_bias = 1 - beta ** (i + 1)
        new_vt = vt / new_bias
        ma.append(new_vt)
    return ma
