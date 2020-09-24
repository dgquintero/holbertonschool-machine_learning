#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as K
import datetime
from tensorflow.keras.datasets import cifar10

def preprocess_data(X, Y):
    """
    Function that pre-processes the data for your model
    Arguments:
        X: ndarray shape (m, 32, 32, 3) containing the CIFAR 10 data
        Y: ndarray hsape (m,) containing the CIFAR 10 labels
    Returns: X_p: preprocessed X
             Y_p: preprocessed Y
    """
