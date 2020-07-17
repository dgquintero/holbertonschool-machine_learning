#!/usr/bin/env python3
"""dotproduct with 2 matrix"""


def mat_mul(mat1, mat2):
    """dotproduct with 2 matrix"""
    # size of columns and rows
    mat1m = len(mat1)
    mat1n = len(mat1[0])
    mat2m = len(mat2)
    mat2n = len(mat2[0])
    a = [row[:] for row in mat1]
    b = [row[:] for row in mat2]
    result = []
    if (mat1n == mat2m):
        for i in range(mat1m):
            row = [0] * mat2n
            result.append(row)
        for i in range(len(result)):
            # iterate through columns
            for j in range(len(result[0])):
                # iterating by rows of mat2
                for k in range(mat1n):
                    result[i][j] += a[i][k] * b[k][j]
        return result
    else:
        return None
