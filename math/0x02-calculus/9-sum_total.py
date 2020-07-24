#!/usr/bin/env python3
"""function def summation_i_squared"""


def summation_i_squared(n):
    """function def summation_i_squared"""
    # if (n == 0):
    #    return 0
    # elif type(n) is int:
    # elif isinstance(n, (int, float)) or n == int(n) or n > 1:
    #    return (int((n ** 2) + summation_i_squared(n - 1)))
    # else:
    #    None
    if type(n) is not int:
        return None
    elif n < 1:
        return None
    else:
        return int((n * (1 + n)) * (2 * n + 1) / 6)
