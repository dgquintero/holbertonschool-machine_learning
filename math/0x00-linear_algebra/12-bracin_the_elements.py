#!/usr/bin/env python3
"""function elementwise"""
import numpy as np


def np_elementwise(mat1, mat2):
    """function elementwise"""
    result = mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
    return result
