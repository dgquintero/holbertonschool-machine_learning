#!/usr/bin/env python3
import numpy as np


def cat_matrices2D(mat1, mat2, axis=0):
    new_array = mat1
    print(axis)
    if axis == 0:
        new_array.append(mat2[0])
    elif axis == 1:
        for i in range(len(mat1[0])):
            new_array[i].append(mat2[i][0])
    # new_array = np.concatenate((mat1, mat2), axis)
    return new_array
