#!/usr/bin/env python3
""" f1_score function"""
import numpy as np


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the f1_score for each class in a confusion matrix
    Arguments:
        confusion: (classes, classes) where row indices represent
            the correct labels and column
        indices represent the predicted labels
    Returns: a ndarray shape(classes,) with the f1_score of each class
    """
    s = sensitivity(confusion)
    p = precision(confusion)
    f1_score = 2 * (p * s) / (p + s)
    return f1_score
