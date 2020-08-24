#!/usr/bin/env python3
""" create_confusion_matrix function"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
     creates a confusion matrix
    Arguments:
        labels: one-hot ndarray shape(m, classes) containing the correct
            labels for each data point
        logists: one-hot shape(m, classes) predicted labels
    Returns: confusion ndarray shape(classes, classes) with row indices
    representing the correct labels and column indices representing the
    predicted labels
    """
    return np.matmul(labels.T, logits)
