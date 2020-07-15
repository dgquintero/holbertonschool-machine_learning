#!/usr/bin/env python3

import numpy as np

def np_elementwise(mat1, mat2):
    new_tuple = []
    new_tuple.append(mat1+mat2)
    new_tuple.append(mat1-mat2)
    new_tuple.append(mat1*mat2)
    new_tuple.append(mat1/mat2)
    return new_tuple
