#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along"""
    # copy of the two matrix
    a = [row[:] for row in mat1]
    b = [row[:] for row in mat2]
    # size of columns and rows
    mat1m = len(mat1)
    mat1n = len(mat1[0])
    mat2m = len(mat2)
    mat2n = len(mat2[0])

    new_array = []
    if (axis == 0) and (mat1n == mat1n):
        new_array = a + b
    elif (axis == 1) and (mat1m == mat2m):
        for i in range(len(mat1[0])):
            new_array.append(a[i] + b[i])
    else:
        return None
    return new_array
