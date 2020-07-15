#!/usr/bin/env python3
"""function add two matrices"""


def add_matrices2D(mat1, mat2):
    """function add two matrices"""
    mat1m = len(mat1)
    mat1n = len(mat1[0])
    mat2m = len(mat2)
    mat2n = len(mat2[0])
    if mat1m == mat2m and mat1n == mat2n:
        result = []
        # iterate through rows
        for i in range(len(mat1)):
            row = []
            # iterate through columns
            for j in range(len(mat1[0])):
                row.append(mat1[i][j] + mat2[i][j])
            result.append(row)
        return result
    else:
        return None
