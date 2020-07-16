#!/usr/bin/env python3
"""function get the shape of a matrix"""


def matrix_shape(matrix):
    """function get the shape of a matrix"""
    mat_shape = []
    if type(matrix[0]) == list:
        mat_shape = [len(matrix), len(matrix[0])]
        if type(matrix[0][0]) == list:
            mat_shape.append(len(matrix[0][0]))
        return mat_shape
    return mat_shape
