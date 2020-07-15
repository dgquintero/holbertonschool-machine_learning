#!/usr/bin/env python3
import numpy as np


def mat_mul(mat1, mat2):
    try:
        #new_array = np.dot(mat1, mat2)
        result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # iterate through rows 
        for i in range(len(mat1)):
            # iterate through columns
            for j in range(len(mat1[0])):
                result[i][j] = mat1[i][j] * mat2[j][i]
        return result
        #return new_array
    except:
        return None
