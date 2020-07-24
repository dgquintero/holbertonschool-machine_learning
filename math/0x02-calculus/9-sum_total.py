#!/usr/bin/env python3
"""function def summation_i_squared"""


def summation_i_squared(n):
    """function def summation_i_squared"""
    if (n == 0):
        return 0
    elif type(n) == int:
        return (n ** 2) + summation_i_squared(n - 1)
    else:
        None
