#!/usr/bin/env python3
import numpy as np

result = [[0, 0, 0], [0, 0, 0]]
def matrix_transpose(matrix):
    result = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return result
