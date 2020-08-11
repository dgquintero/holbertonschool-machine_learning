#!/usr/bin/env python3
"""One-hot encode function"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Arguments:
        Y: numeric classes labels shape(m,)
        classes: the maximum number of classes 
    Return: one-hot enconding Y shape(classes, m)
    """
    # one hot encoder
    onehot_encoder = np.zeros((classes, len(Y)))
    onehot_encoder[Y, np.arange(len(Y))] = 1
    return onehot_encoder

