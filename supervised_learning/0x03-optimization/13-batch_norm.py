#!/usr/bin/env python3
"""batch_norm function"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Function to normalize a batch
    Arguments:
        Z: shape (m, n) that should be normalized
            m: is the number of data points
            n: is the number of features in Z
        gamma: shape (1, n) containing the scales used for
               batch normalization
        beta: shape (1, n) containing the offsets used for
            batch normalization
        epsilon: small number used to avoid division by zero
    Returns: the normalized Z matrix
    """
    mean = Z.mean(axis=0)
    variance = Z.var(axis=0)
    devstd = np.sqrt(variance + epsilon)
    z_centered = Z - mean
    z_normalization = z_centered / devstd
    z_batch_norma = gamma * z_normalization + beta

    return z_batch_norma
