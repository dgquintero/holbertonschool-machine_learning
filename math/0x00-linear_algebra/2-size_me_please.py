#!/usr/bin/env python3
import numpy as np


def matrix_shape(matrix):
    if len(matrix[0][0]) == list:
        mat_shape = [len(matrix), len(matrix[0]), len(matrix[0][0])]
    else:
        mat_shape = [len(matrix), len(matrix[0])]
    return mat_shape
