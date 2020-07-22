#!/usr/bin/env python3


def summation_i_squared(n):
    if type(n) == int:
        sum = 0
        for i in range(n + 1):
            sum += i ** 2
        return sum
    else:
        None
