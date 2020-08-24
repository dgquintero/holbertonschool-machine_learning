#!/usr/bin/env python3
""" sensitivity function"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
    Arguments:
        confusion: (classes, classes) where row indices represent the correct labels and column
        indices represent the predicted labels
    Returns: a ndarray shape(classes,) with the sensitivity of each class
    """
    true_pos = np.diag(confusion)
    false_neg = np.sum(confusion, axis=1) - TP
    sensitivity = true_pos / (true_pos + false_neg)
    return sensitivity
