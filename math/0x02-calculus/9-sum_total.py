#!/usr/bin/env python3
"""function def summation_i_squared"""


def summation_i_squared(n):
    """function def summation_i_squared"""
    # if (n == 0):
    #    return 0
    # elif type(n) is int:
    #    return (int((n ** 2) + summation_i_squared(n - 1)))
    # else:
    #    None
    if type(n) is not int or n < 1:
        return None
    return int((n * (1 + n)) * (2 * n + 1) / 6)
