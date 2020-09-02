#!/usr/bin/env python3
"""function predict"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    function that tests a neural network
    Arguments:
        network: the network model to test
        data: the input data to test the model with
        labels: are the correct one-hot labels of data
        verbose: is a boolean that determines if output
            should be printed during the testing process
    Returns: the prediction for the data
    """
    prediction = network.predict(data, verbose=verbose)
    return prediction
