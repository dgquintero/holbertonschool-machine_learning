#!/usr/bin/env python3
import numpy as np


def mat_mul(mat1, mat2):
    try:
        new_array = np.dot(mat1, mat2)
        return new_array
    except:
        return None
