#!/usr/bin/env python3
"""function def summation_i_squared"""


def summation_i_squared(n):
    """function def summation_i_squared"""
    if type(n) == int:
        sum = 0
        for i in range(n + 1):
            sum += i ** 2
        return sum
    else:
        None
