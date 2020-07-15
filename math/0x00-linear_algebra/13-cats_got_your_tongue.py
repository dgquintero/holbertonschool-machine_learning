#!/usr/bin/env python3
"""function concatenate using numpy"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """function concatenate using numpy"""
    new_array = np.concatenate((mat1, mat2), axis)
    return new_array
