#!/usr/bin/env python3
"""function optimize_model"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Function that converts a label vector into a one-hot matrix
        Arguments:
            labels
            classes
        Returns: the one-hot matrix
    """
    one_hot = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot
