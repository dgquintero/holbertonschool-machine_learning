#!/usr/bin/env python3
import numpy as np

def add_matrices2D(mat1, mat2):
    if np.shape(mat1) == np.shape(mat2):
        sum_array = np.add(mat1, mat2)
        return sum_array

    return None
