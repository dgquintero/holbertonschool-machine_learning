#!/usr/bin/env python3
import numpy as np

def add_matrices2D(mat1, mat2):
    if np.shape(mat1) == np.shape(mat2):
        result = [[0, 0], [0, 0]]
        # iterate through rows 
        for i in range(len(mat1)):
            # iterate through columns
            for j in range(len(mat1[0])):
                result[i][j] = mat1[i][j] + mat2[i][j]
        return result
            
    return None
