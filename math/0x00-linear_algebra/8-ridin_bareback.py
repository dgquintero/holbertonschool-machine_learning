#!/usr/bin/env python3
"""dotproduct with 2 matrix"""

def mat_mul(mat1, mat2):
    """dotproduct with 2 matrix"""
    # size of columns and rows
    mat1m = len(mat1)
    mat1n = len(mat1[0])
    mat2m = len(mat2)
    mat2n = len(mat2[0])
    result = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    # iterate through rows
    if (mat1n == mat2m):
        for i in range(len(mat1)):
            # iterate through columns
            for j in range(len(mat2[0])):
                # iterating by rows of mat2
                for k in range(len(mat2)):
                    result[i][j] += mat1[i][k] * mat2[k][j]
    else:
        result = None
    return result
