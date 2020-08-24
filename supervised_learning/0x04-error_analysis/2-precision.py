#!/usr/bin/env python3
""" precision function"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    Arguments:
        confusion: (classes, classes) where row indices represent
        the correct labels and column
        indices represent the predicted labels
    Returns: a ndarray shape(classes,) with the precision of each class
    """
    true_pos = np.diag(confusion)
    false_neg = np.sum(confusion, axis=1) - true_pos
    false_pos = np.sum(confusion, axis=0) - true_pos

    precision = true_pos / (true_pos + false_pos)
    # PPV = TP/(TP+FP)
    return precision
