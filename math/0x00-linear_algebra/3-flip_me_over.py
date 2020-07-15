#!/usr/bin/env python3
import numpy as np


def matrix_transpose(matrix):
    r = [[0, 0, 0], [0, 0, 0]]
    m = matrix
    r = [[m[j][i] for j in range(len(m))]for i in range(len(m[0]))]
    return r
