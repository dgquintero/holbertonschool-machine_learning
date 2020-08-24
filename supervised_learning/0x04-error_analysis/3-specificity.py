#!/usr/bin/env python3
""" specificity function"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    Arguments:
        confusion: (classes, classes) where row indices represent
        the correct labels and column
        indices represent the predicted labels
    Returns: a ndarray shape(classes,) with the specificity of each class
    """
    true_pos = np.diag(confusion)
    false_neg = np.sum(confusion, axis=1) - true_pos
    false_pos = np.sum(confusion, axis=0) - true_pos
    true_neg = np.sum(confusion) - (true_pos + false_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)
    return specificity
