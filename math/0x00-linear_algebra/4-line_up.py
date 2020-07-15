#!/usr/bin/env python3
import numpy as np


def add_arrays(arr1, arr2):
    if np.shape(arr1) == np.shape(arr2):
        sum_array = [a + b for a, b in zip(arr1, arr2)]
        return sum_array

    return None
