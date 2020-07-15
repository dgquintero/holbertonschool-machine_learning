#!/usr/bin/env python3
"""function that transpose a matrix"""


def matrix_transpose(matrix):
    """matrix transpose function"""
    r = [[0, 0, 0], [0, 0, 0]]
    m = matrix
    r = [[m[j][i] for j in range(len(m))]for i in range(len(m[0]))]
    return r
