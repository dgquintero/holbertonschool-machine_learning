#!/usr/bin/env python3
"""function add two arrays"""


def add_arrays(arr1, arr2):
    """function add two arrays"""
    if len(arr1) == len(arr2):
        sum_array = [a + b for a, b in zip(arr1, arr2)]
        return sum_array

    return None
